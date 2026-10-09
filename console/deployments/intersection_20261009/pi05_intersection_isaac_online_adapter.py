import os
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
os.environ["INTERSECTION_MODEL_ID"] = 'pi05-cup-intersection-30000'
from intersection_adapter import predict
