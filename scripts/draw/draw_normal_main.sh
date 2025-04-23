export CUDA_VISIBLE_DEVICES="0"
nl=3
nh=3
hidden_size=720
mode=main

python demonstration/draw.py \
            --mode $mode \
                --type standard \
                --name acc_map_1 \
               --n_layers $nl\
                --n_heads $nh\
                --hidden_size $hidden_size\
                --model_size normal