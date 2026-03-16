# hrdepth
# NTIRE 2026 - HR Depth from Images of Specular 
# and Transparent Surfaces - Track 1: Stereo

## Method Overview
Our approach combines two models:

1. **RAFT-Stereo** (raftstereo-middlebury pretrained)
   - Stereo disparity estimation
   - Tile-based inference for high-resolution images
   - No scaling factors — full resolution processing

2. **SAM2-Large + LoRA** (fine-tuned on Booster dataset)
   - Detects specular/transparent surfaces
   - Val IoU: 0.8794 after 42 epochs
   - Used to refine disparity in hard regions

## Pipeline
Left Image + Right Image
        ↓
  RAFT-Stereo (tile-based)
        ↓
  Raw Disparity Map
        ↓
  SAM2 detects specular regions
        ↓
  Gaussian blur refinement in specular areas
        ↓
  Final Disparity Map (.npy float32)

## Results
- SAM2 Val IoU: 0.8794
- Training: 42 epochs on Booster training set

## Setup
pip install torch torchvision
pip install git+https://github.com/facebookresearch/sam2.git
pip install albumentations==1.3.1
git clone https://github.com/princeton-vl/RAFT-Stereo

## SAM2 Training
python train.py \
    --data train \
    --ckpt sam2_hiera_large.pt \
    --epochs 60 \
    --batch 1 \
    --lr 3e-4 \
    --lora-rank 4 \
    --out models

## Download Checkpoints
- SAM2-Large: https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt
- Fine-tuned SAM2: [Google Drive link]
- RAFT-Stereo: https://github.com/princeton-vl/RAFT-Stereo

## Dataset
Booster Dataset: https://cvlab-unibo.github.io/booster-web/
