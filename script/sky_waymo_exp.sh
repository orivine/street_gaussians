for s in 006 026 090 105 108 134 150 181; do
  python script/waymo/generate_sky_mask.py \
    --datadir data/waymo/validation/$s \
    --sam_checkpoint grounding_dino_sam_weight/sam_vit_h_4b8939.pth
done