"""Independent matched-precision FEM replay; no model or prescribed nodal motion."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
TUNED = Path('/home/claude/Corl_Track_1/umi_workspace_zhangchi/umi-track1-console-4090-20261009/simulator_profiles/tuned_v1')
sys.path.insert(0,str(TUNED));sys.path.insert(0,str(HERE))
from profiles import PROFILE_IDS, trial_profile
from yubi_isaac_sim_env import env as environment
from yubi_isaac_sim_env import pvc_stiffness, pvc_numerics, run as runner, run_visual_aligned


def main():
    import numpy as np
    gpa = float(os.environ['UMI_PVC_GPA'])
    base = pvc_numerics.precision_profile(pvc_stiffness.profile_for('pvc_shell_e3000mpa_v1'))
    profile = trial_profile(gpa,base)
    assert os.environ['UMI_CUP_MODEL']==profile['id']
    shell = TUNED/'yubi_isaac_sim_env/pvc_shell.py'
    assert hashlib.sha256(shell.read_bytes()).hexdigest()=='2dd9b835ddf2120f7f781086a25a70a47a7df96cdb8d83471c740e5d2e27086b'
    source = Path(os.environ['UMI_MARGIN_SOURCE'])
    assert hashlib.sha256(source.read_bytes()).hexdigest()==os.environ['UMI_MARGIN_SOURCE_SHA256']
    output = Path(os.environ['SIM_ADAPTER_AUDIT_DIR'])
    old_profile_for,old_hz = pvc_stiffness.profile_for,pvc_numerics.physics_hz_for
    pvc_stiffness.SHELL_PROFILE_IDS = (*pvc_stiffness.SHELL_PROFILE_IDS,*PROFILE_IDS)
    pvc_stiffness.profile_for = lambda key,base=None: dict(profile) if key==profile['id'] else old_profile_for(key,base)
    pvc_numerics.physics_hz_for = lambda key: 240 if key in PROFILE_IDS else old_hz(key)
    provenance = dict(classification='frozen_model_action_fem_physics_comparison_not_online_inference',
        source_run='c3b23931ada5',source_sha256=os.environ['UMI_MARGIN_SOURCE_SHA256'],
        original_world_pose_and_jaw_commands_unchanged=True,model_inference_requests=0,
        right_extra_closure_fraction=0.,opening_deadband_enabled=False,increased_friction_enabled=False,
        force_telemetry_available=False,normal_contact_force_abort_available=False,
        unchanged_jaw_drive_limit_Nm=2.05,shape_abort_bound_m=.015,
        no_hidden_rigid_support=True,no_attachments=True,no_prescribed_nodal_motion=True,
        measured_material=False,physics_hz=240,solver_position_iterations=128,
        comparison_to_rigid_changes_contact_backend_and_numerics_not_material_only=True,
        files_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [Path(__file__),HERE/'profiles.py',shell,TUNED/'yubi_isaac_sim_env/pvc_numerics.py']})
    original_write, original_create = runner._write_json,runner.create_sim
    original_observe = environment.DualFrankaYubiCupPlateEnv.observe
    readback = {};last_resource_check = 0.;latest_nodes = None;latest_action = None;saved = set()

    def stop(reason, **details):
        output.mkdir(parents=True,exist_ok=True)
        file = output/'safety_abort.json'
        if not file.exists():
            with file.open('x') as stream:json.dump(dict(reason=reason,**details),stream,indent=2)
        raise RuntimeError('FEM diagnostic safety stop: '+reason)

    def observe(self):
        nonlocal last_resource_check,latest_nodes,latest_action
        if time.monotonic()-last_resource_check>10:
            last_resource_check=time.monotonic()
            pids=subprocess.check_output(['nvidia-smi','-i','0','--query-compute-apps=pid',
                '--format=csv,noheader'],text=True,timeout=5).split()
            if any(p!=str(os.getpid()) for p in pids):stop('other_compute_process_on_gpu0')
        try:state=original_observe(self)
        except RuntimeError as error:
            if 'PVC shell exceeds diagnostic' in str(error):
                stop('shape_change_over_15mm',action_index=self.policy_steps)
            raise
        latest_nodes=self.objects['cup'].nodes();latest_action=self.policy_steps
        if not np.isfinite(latest_nodes).all():stop('nonfinite_fem_nodes')
        if latest_action%90==0 and latest_action not in saved:
            np.savez_compressed(output/f'nodes_{latest_action:04d}.npz',nodes=latest_nodes)
            saved.add(latest_action)
        return state

    def create(*args,**kwargs):
        app,env=original_create(*args,**kwargs)
        from yubi_isaac_sim_env.pvc_response_probe import material_readback
        from pxr import UsdPhysics
        assert env.deformable_cup and env.task_config['physics_hz']==240
        assert not env.audit_cup_contacts,'Rigid contact telemetry cannot measure this FEM cup'
        assert not env.stage.GetPrimAtPath('/World/Objects/Cup').HasAPI(UsdPhysics.RigidBodyAPI)
        assert not env.stage.GetPrimAtPath('/World/Objects/Cup/Collision').IsActive()
        assert not env.stage.GetPrimAtPath('/World/Objects/Cup/Visual').IsActive()
        for side in ('LeftMount','RightMount'):
            for joint in ('yubi_finger_joint','yubi_finger_mimic_joint'):
                p=env.stage.GetPrimAtPath(f'/World/Robots/{side}/Panda/joints/{joint}')
                assert abs(p.GetAttribute('drive:angular:physics:maxForce').Get()-2.05)<1e-6
        readback.update(material_readback(env.stage,profile))
        return app,env

    def write(path,value):
        if Path(path).name in ('report.json','manifest.json'):
            value['frozen_pvc_diagnostic']=dict(provenance,material_readback=readback)
            value['evaluation_class']=provenance['classification']
            value['model_inference_requests']=0
        return original_write(path,value)

    environment.DualFrankaYubiCupPlateEnv.observe=observe
    runner.create_sim,runner._write_json=create,write
    try:return run_visual_aligned.main()
    finally:
        if latest_nodes is not None:
            np.savez_compressed(output/'final_nodes.npz',nodes=latest_nodes,action_index=latest_action)


if __name__=='__main__':raise SystemExit(main())
