"""Reuse tested layer offload in this isolated intersection process only."""
import inspect
import os
from pathlib import Path
import socket
import sys

ROOT=Path('/home/claude/workspace/umi_cup_intersection_models_4090_20261009')

def main():
    if socket.gethostname()!='benyun-workstation' or os.environ.get('CUDA_VISIBLE_DEVICES')!='1':
        raise RuntimeError('Expected private RTX4090 GPU1 launcher')
    source=ROOT/'source/OpenWAM'
    sys.path.insert(0,str(source))
    import torch
    from openwam.model.video_backbone.wan_backbone import WanBase
    from openwam.model.video_backbone.wan import encode as wan_encode
    from openwam.model.video_backbone.wan.shared.models.model_loader import ModelPool
    from openwam.model.video_backbone.wan.shared.core.vram.layers import enable_vram_management
    from openwam_cpu_text_encoder_4090 import _patch_scope
    if Path(inspect.getfile(WanBase)).resolve()!=source/'openwam/model/video_backbone/wan_backbone.py':
        raise RuntimeError('Unexpected intersection OpenWAM source')
    if 'RTX4090' not in torch.cuda.get_device_name(0).replace(' ','').upper():
        raise RuntimeError('Not an RTX4090')
    from intersection_server import main as serve
    with _patch_scope(torch,WanBase,wan_encode,ModelPool,enable_vram_management):
        serve()

if __name__=='__main__':
    main()
