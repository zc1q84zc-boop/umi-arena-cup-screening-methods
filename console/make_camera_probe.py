"""Capture the actual three model inputs without contacting a model server."""
import json
import os
from pathlib import Path
import numpy as np
from PIL import Image

def act(observation, step, episode):
    root = Path(os.environ['SIM_ADAPTER_AUDIT_DIR'])
    root.mkdir(parents=True, exist_ok=True)
    if step == 0:
        for name, rgb in observation['images'].items():
            image = np.asarray(rgb)
            assert image.shape == (480, 640, 3) and image.dtype == np.uint8
            Image.fromarray(image).save(root / (name + '.png'))
        (root / 'camera_probe.json').write_text(json.dumps({
            'images': observation.get('image_metadata'), 'robots': observation['robots'],
            'purpose': 'framing inspection, not physical calibration or task evaluation'
        }, indent=2))
    return {}
