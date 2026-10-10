"""Render all active cameras together and reuse a sampled physical state."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np


class SharedCameraRender:
    def __init__(self, runner):
        self.runner = runner
        self.original_rgb = runner._rendered_rgb
        self.original_sample = runner._sample
        self.original_policy = runner._policy_observation
        self.context = None
        self.prepared = None
        self.last = None
        self.audit = None
        self.stats = dict(profile='shared_camera_render_v1', groups_rendered=0,
                          render_calls=0, reused_policy_groups=0, policy_groups=0,
                          recorded_groups=0, render_wall_s=0., physics_changed=False)

    def _signature(self, cameras, specs):
        return {name: (id(camera), tuple(specs[name]['resolution']),
                       bool(specs[name].get('horizontal_flip_for_dataset', False)),
                       tuple(self.runner._cpu_array(camera.get_world_pose(camera_axes='usd')[0])),
                       tuple(self.runner._cpu_array(camera.get_world_pose(camera_axes='usd')[1])))
                for name, camera in cameras.items()}

    @staticmethod
    def _same_signature(a, b):
        if a.keys() != b.keys():
            return False
        for name in a:
            if a[name][:3] != b[name][:3]:
                return False
            if not np.allclose(a[name][3], b[name][3], atol=1e-7, rtol=0):
                return False
            qa, qb = np.asarray(a[name][4]), np.asarray(b[name][4])
            if min(np.linalg.norm(qa-qb), np.linalg.norm(qa+qb)) > 1e-7:
                return False
        return True

    def _prepare(self):
        event, cameras, specs, world, state = self.context
        stamp = float(state['physics_time_s'])
        if not math.isclose(float(world.current_time), stamp, abs_tol=1e-6, rel_tol=0):
            raise RuntimeError('Camera request does not match the current physics state')
        signature = self._signature(cameras, specs)
        if (self.last is not None and self.last['world'] is world
                and self.last['physics_time_s'] == stamp
                and self._same_signature(self.last['signature'], signature)):
            self.prepared = {**self.last, 'reused': True}
            return

        # Collect each rendered frame rather than retaining a rate-limited sensor frame.
        for camera in cameras.values():
            if camera.get_frequency() != -1:
                camera.set_frequency(-1)
        started = time.perf_counter()
        calls = 0

        def render():
            nonlocal calls
            world.render()
            calls += 1
            self.stats['render_calls'] += 1
            if not math.isclose(float(world.current_time), stamp, abs_tol=1e-6, rel_tol=0):
                raise RuntimeError('Rendering unexpectedly advanced physics')

        # Keep the existing five-frame RTX warmup, shared by every render product.
        for _ in range(5):
            render()
        images, metadata = {}, {}
        for _ in range(24):
            render()
            images, metadata = {}, {}
            for name, camera in cameras.items():
                spec = specs[name]
                rgb = camera.get_rgb(device='cpu')
                frame = camera.get_current_frame()
                frame_time = frame.get('rendering_time') if frame else None
                if (rgb is None or frame_time is None
                        or not math.isclose(float(frame_time), stamp, abs_tol=1e-5, rel_tol=0)):
                    break
                array = np.asarray(rgb)
                expected = (spec['resolution'][1], spec['resolution'][0], 3)
                if (array.shape != expected or not np.isfinite(array).all()
                        or float(array.max())-float(array.min()) < 2):
                    break
                if spec.get('horizontal_flip_for_dataset', False):
                    array = array[:, ::-1]
                owned = self.runner._policy_rgb(array, spec)
                owned.setflags(write=False)
                images[name] = owned
                position, orientation = camera.get_world_pose(camera_axes='usd')
                metadata[name] = {'rendering_time_s': float(frame_time), 'pose_world': {
                    'position_m': self.runner._cpu_array(position).tolist(),
                    'quaternion_wxyz': self.runner._cpu_array(orientation).tolist()}}
            if len(images) == len(cameras):
                break
        else:
            diagnostics = {}
            for name, camera in cameras.items():
                frame = camera.get_current_frame() or {}
                rgb = camera.get_rgb(device='cpu')
                diagnostics[name] = dict(
                    rendering_time=frame.get('rendering_time'),
                    rendering_frame=frame.get('rendering_frame'),
                    shape=None if rgb is None else list(rgb.shape))
            raise RuntimeError('All cameras must provide valid RGB at the requested physics time: '
                               + str(dict(physics_time=stamp, cameras=diagnostics)))
        elapsed = time.perf_counter()-started
        self.stats['groups_rendered'] += 1
        self.stats['render_wall_s'] += elapsed
        self.last = dict(world=world, signature=signature, physics_time_s=stamp,
                         group=self.stats['groups_rendered'], images=images,
                         metadata=metadata, render_calls=calls,
                         image_sha256={name: hashlib.sha256(value.tobytes()).hexdigest()
                                       for name, value in images.items()})
        self.prepared = {**self.last, 'reused': False}

    def rgb(self, camera, world, spec):
        if self.context is None:
            return self.original_rgb(camera, world, spec)
        if self.prepared is None:
            self._prepare()
        if self.context[3] is not world or self.context[1].get(spec['name']) is not camera:
            raise RuntimeError('Camera is outside the active shared render group')
        return self.prepared['images'][spec['name']]

    def _finish(self):
        if self.prepared is None:
            return
        event = self.context[0]
        self.stats['policy_groups' if event == 'policy' else 'recorded_groups'] += 1
        if event == 'policy' and self.prepared['reused']:
            self.stats['reused_policy_groups'] += 1
        folder = os.environ.get('SIM_ADAPTER_AUDIT_DIR')
        if folder:
            if self.audit is None:
                self.audit = (Path(folder)/'shared_camera_render.jsonl').open('x')
            self.audit.write(json.dumps(dict(event=event,
                physics_time_s=self.prepared['physics_time_s'], group=self.prepared['group'],
                reused=self.prepared['reused'],
                render_calls=0 if self.prepared['reused'] else self.prepared['render_calls'],
                image_sha256=self.prepared['image_sha256'],
                rendering_times_s={n: m['rendering_time_s'] for n,m in self.prepared['metadata'].items()}))+'\n')
            self.audit.flush()

    def sample(self, video, joints, camera, world, spec, episode, sample_index,
               phase, policy_step, observation, extra_videos=None, cameras=None,
               specs=None, camera_audit=None):
        if phase == 'reset':
            self.last = None
        self.context = ('record', cameras or {spec['name']:camera},
                        specs or {spec['name']:spec}, world, observation)
        self.prepared = None
        try:
            result = self.original_sample(video, joints, camera, world, spec, episode,
                sample_index, phase, policy_step, observation, extra_videos,
                cameras, specs, camera_audit)
            self._finish()
            return result
        finally:
            self.context = self.prepared = None

    def policy(self, state, cameras, specs, world):
        self.context = ('policy', cameras, specs, world, state)
        self.prepared = None
        try:
            result = self.original_policy(state, cameras, specs, world)
            for name, metadata in result['image_metadata'].items():
                metadata.update(self.prepared['metadata'][name])
                metadata['shared_render_group'] = self.prepared['group']
                metadata['shared_render_reused'] = self.prepared['reused']
            self._finish()
            return result
        finally:
            self.context = self.prepared = None

    def close(self):
        if self.audit is not None:
            self.audit.close()

    def uninstall(self):
        self.close()
        self.runner._rendered_rgb = self.original_rgb
        self.runner._sample = self.original_sample
        self.runner._policy_observation = self.original_policy


def install_shared_camera_render(runner):
    shared = SharedCameraRender(runner)
    runner._rendered_rgb = shared.rgb
    runner._sample = shared.sample
    runner._policy_observation = shared.policy
    return shared
