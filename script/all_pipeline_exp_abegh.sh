#!/bin/bash
for s in 006 026 090 105 108 134 150 181; do
  echo "===== Start scene ${s} ====="
  python train.py --config configs/experiments_waymo/waymo_val_${s}_test.yaml \
  gpus '[3]' \
  task waymo_exp_h \
  exp_name waymo_val_${s}_abegh \
  method.priority.source 'box_residual' \
  method.priority.residual_scope 'box' \
  method.priority.residual_lambda 1.0 \
  method.priority.residual_blend 0.5 \
  method.priority.residual_percentile 0.95 \
  method.photo_loss.mode 'priority_l1' \
  method.sampler.mode uniform \
  method.density.mode 'baseline' \
  method.motion.mode 'emd_pose_lite' \
  method.motion.time_encoding learnable \
  method.motion.delta_t_scale 0.05 \
  method.motion.delta_r_scale 0.03 \
  method.motion.warmup_start 2000 \
  method.motion.warmup_end 8000 \
  method.motion.lambda_motion_reg_t 0.001 \
  method.motion.lambda_motion_reg_r 0.001 \
  method.object_zone.mode 'box_l1_dssim' \
  method.object_zone.source 'box' \
  logging.wandb.enabled False
  python render.py --config configs/experiments_waymo/waymo_val_${s}_test.yaml mode evaluate \
  gpus '[3]' \
  task waymo_exp_h \
  exp_name waymo_val_${s}_abegh \
  method.motion.mode 'emd_pose_lite'
  python render.py --config configs/experiments_waymo/waymo_val_${s}_test.yaml mode trajectory \
  gpus '[3]' \
  task waymo_exp_h \
  exp_name waymo_val_${s}_abegh \
  method.motion.mode 'emd_pose_lite'
  python metrics_complete.py --config configs/experiments_waymo/waymo_val_${s}_test.yaml \
  gpus '[3]' \
  task waymo_exp_h \
  exp_name waymo_val_${s}_abegh
  echo "===== Finish scene ${s} ====="
done
