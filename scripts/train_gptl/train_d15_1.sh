export CUDA_VISIBLE_DEVICES="3"

num_icl_train_traces=10000
num_mk_train_traces=20000
cl=1
cu=6
train_epoch=14
tbs=4
ebs=4
save_steps=20000

nl=36
nh=20
hidden_size=1280

for len in 15
do
for gt in 0 1 2 3 4
do
for child_len in 7 6 4
do
python main.py --num_icl_train_traces $num_icl_train_traces \
               --num_mk_train_traces  $num_mk_train_traces\
               --graph_type $gt\
               --graph_len $len\
               --max_child_chain_len $child_len\
                --context_lower $cl\
                --context_upper $cu\
                --context_div $cu\
               --n_layers $nl\
                --n_heads $nh\
                --hidden_size $hidden_size\
                --if_train y\
                --train_epoch $train_epoch \
                --per_device_train_batch_size $tbs\
                --per_device_eval_batch_size $ebs\
                --save_steps $save_steps\
                --if_test y\
                --if_plot n \
                --if_probe n\

done
done
done
