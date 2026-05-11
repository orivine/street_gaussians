#!/bin/bash
for s in 006 026 090 105 108 134 150 181; do
  echo "===== Start scene ${s} ====="
  python train.py --config configs/experiments_waymo/waymo_val_${s}_test.yaml \
  gpus '[1]' \
  task waymo_exp_abecd1_ablation \
  exp_name waymo_val_${s}_abe_box_EMD \
  method.priority.source box_residual \
  method.priority.residual_scope 'box' \
  method.photo_loss.mode priority_l1 \
  method.sampler.mode uniform \
  method.density.mode baseline \
  method.motion.mode emd_pose_lite \
  logging.wandb.enabled False
  python render.py --config configs/experiments_waymo/waymo_val_${s}_test.yaml mode evaluate \
  gpus '[1]' \
  task waymo_exp_abecd1_ablation \
  exp_name waymo_val_${s}_abe_box_EMD \
  method.motion.mode emd_pose_lite
  python render.py --config configs/experiments_waymo/waymo_val_${s}_test.yaml mode trajectory \
  gpus '[1]' \
  task waymo_exp_abecd1_ablation \
  exp_name waymo_val_${s}_abe_box_EMD \
  method.motion.mode emd_pose_lite
  python metrics_complete.py --config configs/experiments_waymo/waymo_val_${s}_test.yaml \
  gpus '[1]' \
  task waymo_exp_abecd1_ablation \
  exp_name waymo_val_${s}_abe_box_EMD
  echo "===== Finish scene ${s} ====="
done
