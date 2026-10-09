"""The common UMI adapter, restricted to verified official pretrained weights."""
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ['LINGBOT_EXPECTED_MODEL_ID'] = 'lingbot-vla2-official-pretrained'
from lingbot_isaac_online_adapter import predict  # noqa: E402,F401
