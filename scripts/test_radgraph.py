import sys
from pathlib import Path
import os
import torch


try:
    from radgraph import RadGraph, F1RadGraph
    print("RadGraph classes imported successfully.")
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)

def test_inference():
    try:
        # Initialize RadGraph
        rg = RadGraph(model_type="radgraph", cuda=0 if torch.cuda.is_available() else -1)
        print("RadGraph initialized.")
        
        sample_text = ["The lungs are clear without focal consolidation."]
        annotations = rg(sample_text)
        print(f"Inference successful. Captured {len(annotations['0']['entities'])} entities.")
        
    except Exception as e:
        print(f"Inference Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_inference()
