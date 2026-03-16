"""
pipeline.py — Complete NTIRE 2026 Stereo Depth Estimation Pipeline
Combines RAFT-Stereo + SAM2 for high-resolution disparity estimation
on specular and transparent surfaces.
"""

import sys
import cv2
import numpy as np
import torch
import zipfile
import io
import argparse
from pathlib import Path
from tqdm import tqdm
from argparse import Namespace

# ── RAFT-Stereo Setup ─────────────────────────────────────────────────────────
def load_raft_stereo(ckpt_path):
    sys.path.append('RAFT-Stereo')
    sys.path.append('RAFT-Stereo/core')
    from raft_stereo import RAFTStereo

    args = Namespace(
        corr_implementation='alt',
        corr_levels=4, corr_radius=4,
        dropout=0.0, hidden_dims=[128]*3, context_dims=[128]*3,
        context_norm='batch', mixed_precision=True, n_downsample=2,
        n_gru_layers=3, shared_backbone=False, slow_fast_gru=True,
        valid_iters=7
    )
    model = RAFTStereo(args)
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    new_ckpt = {k.replace('module.', ''): v for k, v in ckpt.items()}
    model.load_state_dict(new_ckpt)
    model = model.cuda()
    model.eval()
    print("✅ RAFT-Stereo loaded!")
    return model

# ── SAM2 Setup ────────────────────────────────────────────────────────────────
def load_sam2(ckpt_path):
    import sam2 as _sam2_pkg
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from hydra.utils import instantiate

    config_dir = str(
        Path(_sam2_pkg.__file__).parent / "configs" / "sam2"
    )
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=config_dir, version_base="1.2"):
        cfg = compose(config_name="sam2_hiera_l")

    model = instantiate(cfg.model)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state, strict=False)
    model = model.cuda()
    model.eval()
    print("✅ SAM2 loaded!")
    return model

# ── RAFT-Stereo Inference (tile-based for high-res images) ───────────────────
def estimate_disparity(model, left_path, right_path,
                       orig_h=3008, orig_w=4112,
                       tile_h=752, tile_w=1024, overlap=64):
    left_img  = cv2.cvtColor(cv2.imread(str(left_path)),  cv2.COLOR_BGR2RGB)
    right_img = cv2.cvtColor(cv2.imread(str(right_path)), cv2.COLOR_BGR2RGB)

    disparity = np.zeros((orig_h, orig_w), dtype=np.float32)
    count     = np.zeros((orig_h, orig_w), dtype=np.float32)

    y_steps = range(0, orig_h - tile_h + 1, tile_h - overlap)
    x_steps = range(0, orig_w - tile_w + 1, tile_w - overlap)

    with torch.no_grad():
        for y in y_steps:
            for x in x_steps:
                torch.cuda.empty_cache()
                y2 = min(y + tile_h, orig_h)
                x2 = min(x + tile_w, orig_w)
                y1, x1 = y2 - tile_h, x2 - tile_w

                lt = torch.tensor(
                    left_img[y1:y2, x1:x2]
                ).permute(2,0,1).float()[None].cuda()
                rt = torch.tensor(
                    right_img[y1:y2, x1:x2]
                ).permute(2,0,1).float()[None].cuda()

                _, disp = model(lt, rt, iters=7, test_mode=True)
                disparity[y1:y2, x1:x2] += disp.squeeze().cpu().numpy()
                count[y1:y2, x1:x2] += 1

    disparity = disparity / np.maximum(count, 1)
    return np.abs(disparity).astype(np.float32)

# ── SAM2 Mask Inference ───────────────────────────────────────────────────────
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

@torch.no_grad()
def get_sam2_mask(model, img_path, orig_h=3008, orig_w=4112):
    img = cv2.imread(str(img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (1024, 1024), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    img = (img - MEAN) / STD
    tensor = torch.tensor(img).permute(2,0,1).unsqueeze(0).float().cuda()

    dec = model.sam_mask_decoder
    backbone_out = model.image_encoder(tensor)
    fpn = backbone_out["backbone_fpn"]
    feat_s1 = dec.conv_s1(fpn[1])
    feat_s0 = dec.conv_s0(fpn[0])
    sparse_emb, dense_emb = model.sam_prompt_encoder(
        points=None, boxes=None, masks=None
    )
    low_res_masks, _, _, _ = dec(
        image_embeddings=backbone_out["vision_features"],
        image_pe=model.sam_prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_emb,
        dense_prompt_embeddings=dense_emb,
        multimask_output=False,
        repeat_image=False,
        high_res_features=[feat_s0, feat_s1],
    )
    mask = torch.sigmoid(low_res_masks[0, 0]).cpu().numpy()
    mask = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    return (mask > 0.5).astype(np.float32)

# ── Combine disparity + SAM2 mask ────────────────────────────────────────────
def refine_disparity(disp, mask):
    disp_smooth = cv2.GaussianBlur(disp, (7, 7), 2.0)
    return (disp * (1 - mask) + disp_smooth * mask).astype(np.float32)

# ── Main Pipeline ─────────────────────────────────────────────────────────────
def run(args):
    raft  = load_raft_stereo(args.raft_ckpt)
    sam2  = load_sam2(args.sam2_ckpt)

    test_root = Path(args.data)
    scenes    = sorted([d for d in test_root.iterdir() if d.is_dir()])
    print(f"Found {len(scenes)} scenes")

    zip_path = Path(args.output)
    with zipfile.ZipFile(zip_path, 'w',
                         zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for scene in tqdm(scenes):
            left_imgs  = sorted((scene/'left').glob('*.png'))
            right_imgs = sorted((scene/'right').glob('*.png'))

            for lp, rp in zip(left_imgs, right_imgs):
                # Step 1: RAFT-Stereo disparity
                disp = estimate_disparity(raft, lp, rp)

                # Step 2: SAM2 mask
                torch.cuda.empty_cache()
                mask = get_sam2_mask(sam2, lp)

                # Step 3: Refine
                disp_final = refine_disparity(disp, mask)

                # Step 4: Save to zip
                buf = io.BytesIO()
                np.save(buf, disp_final)
                zf.writestr(
                    f"{scene.name}/{lp.stem}.npy",
                    buf.getvalue()
                )

            print(f"✅ {scene.name}")

    size = zip_path.stat().st_size / 1024/1024/1024
    print(f"\n✅ {zip_path.name} — {size:.2f} GB")
    print("Ready to submit!")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data",      default="test_stereo_nogt")
    p.add_argument("--raft_ckpt", default="RAFT-Stereo/models/raftstereo-middlebury.pth")
    p.add_argument("--sam2_ckpt", default="models/sam2_best.pth")
    p.add_argument("--output",    default="submission.zip")
    run(p.parse_args())
