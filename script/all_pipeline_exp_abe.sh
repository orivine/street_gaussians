# !/bin/bash
for s in 006 026 090 105 108 134 150 181; do
  python train.py --config configs/experiments_waymo/waymo_val_${s}_test.yaml \
  gpus '[0]' \
  exp_name waymo_val_${s}_abe \
  method.priority.source box_residual \
  method.priority.residual_scope 'box' \
  method.photo_loss.mode priority_l1 \
  method.sampler.mode uniform \
  method.density.mode baseline \
  logging.wandb.enabled True
  python render.py --config configs/experiments_waymo/waymo_val_${s}_test.yaml mode evaluate \
  gpus '[0]' \
  exp_name waymo_val_${s}_abe
  python render.py --config configs/experiments_waymo/waymo_val_${s}_test.yaml mode trajectory \
  gpus '[0]' \
  exp_name waymo_val_${s}_abe
  python metrics_complete.py --config configs/experiments_waymo/waymo_val_${s}_test.yaml \
  gpus '[0]' \
  exp_name waymo_val_${s}_abe
done
