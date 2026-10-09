import os
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
os.environ["INTERSECTION_MODEL_ID"] = 'openwam-cup-intersection-fullpass-5069'
from intersection_adapter import predict
