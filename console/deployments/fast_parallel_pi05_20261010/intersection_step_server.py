"""Official-input inference for future-aligned 10 Hz intersection checkpoints."""
import argparse
import base64
from http.server import BaseHTTPRequestHandler,HTTPServer
from io import BytesIO
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
from official_cup_prompts import validate_instruction,PROMPT_SOURCE,PROMPT_PROTOCOL

TIMING={'action_hz':10,'pose_rows':[0],'gripper_rows':[0],
        'basis':'training row0 already composes future body deltas t+1..t+3; jaw target t+2'}


def observation(payload):
    assert set(payload)=={'left_jpeg','right_jpeg','relative_pose_xyzw','gripper_rad','prompt','episode','step'}
    assert type(payload['step']) is int and payload['step']>=0
    assert type(payload['episode']) is int and payload['episode']>=0
    pose=np.asarray(payload['relative_pose_xyzw'],dtype=np.float32)
    jaws=np.asarray(payload['gripper_rad'],dtype=np.float32)
    assert pose.shape==(7,) and jaws.shape==(2,) and np.isfinite(pose).all() and np.isfinite(jaws).all()
    assert abs(np.linalg.norm(pose[3:])-1)<.005
    value={'observation.pose.left_hand_root_to_right_hand_root.absolute':pose,
           'observation.joint_states':jaws,'prompt':validate_instruction(payload['prompt'])}
    for side in ('left','right'):
        raw=base64.b64decode(payload[side+'_jpeg'],validate=True)
        assert len(raw)<=2_000_000
        with Image.open(BytesIO(raw)) as image:rgb=np.asarray(image.convert('RGB'),dtype=np.uint8)
        assert rgb.shape==(480,640,3)
        value['observation.image.'+side]=rgb
    return value


class Pi05:
    def __init__(self,args):
        sys.path[:0]=[str(args.evaluation_root),str(args.openpi_root/'src')]
        try:
            import lerobot.common.datasets.lerobot_dataset
        except ModuleNotFoundError as exc:
            if not (exc.name or '').startswith('lerobot.common'):raise
            # Same compatibility mapping used by this checkpoint's trainer.
            import types
            import lerobot
            import lerobot.datasets.lerobot_dataset as current
            common=types.ModuleType('lerobot.common');datasets=types.ModuleType('lerobot.common.datasets')
            common.datasets=datasets;datasets.lerobot_dataset=current;lerobot.common=common
            sys.modules.update({'lerobot.common':common,'lerobot.common.datasets':datasets,
                'lerobot.common.datasets.lerobot_dataset':current})
        from openpi.models.pi0_config import Pi0Config
        from openpi.training.config import TrainConfig,AssetsConfig
        from umi_arena.yubi.config import LeRobotYubiDataConfig
        from openpi.policies import policy_config
        marker=json.loads((args.checkpoint/'inference_export.json').read_text())
        assert marker['model']==args.model_id and marker['asset_id']=='intersection_v2' and marker['action_hz']==10
        cfg=TrainConfig(name='pi05_cup_intersection',model=Pi0Config(pi05=True,action_dim=32,action_horizon=32),
            data=LeRobotYubiDataConfig(repo_id='yubi-cup-intersection-independent-v2',assets=AssetsConfig(asset_id='intersection_v2')))
        self.policy=policy_config.create_trained_policy(cfg,args.checkpoint)

    def infer(self,obs):
        values=np.asarray(self.policy.infer(obs)['actions'],dtype=np.float32)
        assert values.shape==(32,16) and np.isfinite(values).all()
        # The new reader already aligns both labels to the same future slot.
        return values


class OpenWAM:
    def __init__(self,args):
        sys.path.insert(0,str(args.openwam_root))
        import setuptools
        from omegaconf import OmegaConf
        from openwam.deploy import model_loader
        from openwam.deploy.server import build_server_from_config
        import importlib.util
        spec=importlib.util.spec_from_file_location('saved_contract',args.checkpoint/'official_contract/contract.py')
        contract=importlib.util.module_from_spec(spec);spec.loader.exec_module(contract)
        self.contract=contract
        norms=json.loads((args.checkpoint/'official_contract/normalization.json').read_text())
        saved=np.load(args.checkpoint/'normalization_stats.npy',allow_pickle=True).item()
        assert set(saved)=={'state','action'}
        for kind in ('state','action'):
            for key in norms[kind]:assert np.allclose(saved[kind][key],norms[kind][key],atol=1e-6)
        class Normalizer:
            def normalize_proprio(self,raw):
                values=np.asarray(raw,dtype=np.float32)
                assert values.shape[-1]==9 and np.isfinite(values).all()
                out=np.zeros((*values.shape[:-1],80),dtype=np.float32)
                out[...,:9]=contract.normalize(values,norms['state'])
                return out
            normalize=normalize_proprio
            def unnormalize(self,unified):return contract.actions_to_wire(unified,norms['action'])
        normalizer=Normalizer()
        def build_normalizer(cfg,ckpt_dir):
            dl=cfg.dataloader
            assert Path(ckpt_dir).resolve()==args.checkpoint.resolve()
            assert dl.type=='umi_cup_merged_official' and dl.normalize_mode=='min-max'
            assert dl.unify_action and list(dl.unify_action_map)==['0-15']
            assert (dl.height,dl.width)==(256,640)
            return normalizer
        model_loader._build_normalizer=build_normalizer
        cfg=OmegaConf.load(args.openwam_root/'configs/deploy.yaml')
        OmegaConf.update(cfg,'optimization.compile.enabled',False)
        OmegaConf.update(cfg,'optimization.decode_video',False)
        OmegaConf.update(cfg,'optimization.dit_cache.enabled',False)
        self.engine=build_server_from_config(cfg,str(args.checkpoint),device=args.device,
            ckpt_name='checkpoint_step_5069.safetensors').engine

    def infer(self,obs):
        image,state,prompt=self.contract.observation(obs)
        result=self.engine.generate({'first_frame_image':[image],'proprio':state,'prompt':prompt,
            'num_frames':33,'video_num_frames':9,'height':256,'width':640,'seed':42})
        values=np.asarray(result['actions'],dtype=np.float32)
        assert values.shape==(32,16) and np.isfinite(values).all()
        return values


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--family',choices=['pi05','openwam'],required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--model-id',required=True)
    p.add_argument('--port',type=int,required=True)
    p.add_argument('--evaluation-root',type=Path)
    p.add_argument('--openpi-root',type=Path)
    p.add_argument('--openwam-root',type=Path)
    p.add_argument('--device',default='cuda')
    args=p.parse_args()
    assert args.family != 'pi05' or args.checkpoint.name in ('10000','20000','30000')
    expected={'pi05':'pi05-cup-intersection-'+args.checkpoint.name,'openwam':'openwam-cup-intersection-fullpass-5069'}
    assert args.model_id==expected[args.family]
    engine=(Pi05 if args.family=='pi05' else OpenWAM)(args)
    class Handler(BaseHTTPRequestHandler):
        def reply(self,status,data):
            raw=json.dumps(data,separators=(',',':')).encode();self.send_response(status)
            self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(raw)))
            self.end_headers();self.wfile.write(raw)
        def do_GET(self):
            if self.path!='/health':self.send_error(404);return
            self.reply(200,{'ready':True,'model':args.model_id,'checkpoint':str(args.checkpoint),
                'action_timing':TIMING,'input_views':['left_wrist','right_wrist']})
        def do_POST(self):
            if self.path!='/infer':self.send_error(404);return
            try:
                size=int(self.headers.get('Content-Length','0'));assert 0<size<=4_000_000
                payload=json.loads(self.rfile.read(size));obs=observation(payload)
                started=time.monotonic();actions=engine.infer(obs)
                self.reply(200,{'model':args.model_id,'episode':payload['episode'],'step':payload['step'],
                    'actions':actions[:1].tolist(),'action_timing':TIMING,'prompt':payload['prompt'],
                    'prompt_source':PROMPT_SOURCE,'prompt_protocol':PROMPT_PROTOCOL,
                    'future_observation_used':False,'latency_ms':(time.monotonic()-started)*1000})
            except Exception as exc:self.reply(400,{'error':f'{type(exc).__name__}: {exc}'})
    print('INTERSECTION_READY '+args.model_id,flush=True)
    HTTPServer(('127.0.0.1',args.port),Handler).serve_forever()


if __name__=='__main__':main()
