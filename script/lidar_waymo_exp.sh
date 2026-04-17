for s in 006 026 090 105 108 134 150 181; do
  python script/waymo/generate_lidar_depth.py --datadir data/waymo/validation/$s
done