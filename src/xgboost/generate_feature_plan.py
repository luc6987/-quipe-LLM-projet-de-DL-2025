"""
Shim to keep CLI compatibility after relocating feature plan generator.
Actual implementation now lives in src/preprocessing/generate_feature_plan.py.
"""
from src.preprocessing.generate_feature_plan import generate_plan

if __name__ == '__main__':
    generate_plan()

