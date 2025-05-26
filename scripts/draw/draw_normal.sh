export CUDA_VISIBLE_DEVICES="0"



parent_dir="data_and_models/normal_and_simpler"

python draw.py \
     --config-name config_normal.yaml\
    draw.parent_dir=$parent_dir\
    draw.mode=main

python draw.py \
     --config-name config_normal.yaml\
    draw.parent_dir=$parent_dir\
    draw.mode=ratio
