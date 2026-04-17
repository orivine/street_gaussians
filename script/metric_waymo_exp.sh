for s in 006 026 090 105 108 134 150 181; do
  python metrics_complete.py --config configs/experiments_waymo/waymo_val_$s.yaml
done