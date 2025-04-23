export CUDA_VISIBLE_DEVICES="4"


num_icl_train_traces=10000
num_mk_train_traces=20000
cl=1
cu=6

nl=12
nh=12
hidden_size=720

python demonstration/draw.py --mode main \
                --type standard \
                --name acc_map_1 \
                --num_icl_train_traces $num_icl_train_traces \
               --num_mk_train_traces  $num_mk_train_traces\
                --context_lower $cl\
                --context_upper $cu\
                --context_div $cu\
               --n_layers $nl\
                --n_heads $nh\
                --hidden_size $hidden_size\
                --model_size gptsmall


##$(($num_traces+$num_traces)) \