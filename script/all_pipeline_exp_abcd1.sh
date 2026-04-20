# !/bin/bash
for s in 006 026 090 105 108 134 150 181; do
  python train.py --config configs/experiments_waymo/waymo_val_${s}_test.yaml gpu '[3]' method.priority.source box method.photo_loss.mode priority_l1 method.sampler.mode priority_mix method.density.mode object exp_name waymo_val_${s}_abcd1
  python render.py --config configs/experiments_waymo/waymo_val_${s}_test.yaml mode evaluate gpu '[3]' exp_name waymo_val_${s}_abcd1
  python render.py --config configs/experiments_waymo/waymo_val_${s}_test.yaml mode trajectory gpu '[3]' exp_name waymo_val_${s}_abcd1
  python metrics_complete.py --config configs/experiments_waymo/waymo_val_${s}_test.yaml gpu '[3]' exp_name waymo_val_${s}_abcd1
done
