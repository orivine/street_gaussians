python script/waymo/waymo_converter.py \
    --root_dir data/waymo/raw_validation \
    --save_dir data/waymo/validation \
    --split_file script/waymo/waymo_splits/val_dynamic.txt \
    --segment_file script/waymo/waymo_splits/segment_list_val.txt \
    --track_file data/waymo/tracker/result.json