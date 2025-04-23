export CUDA_VISIBLE_DEVICES="4"

train_epoch=14
save_steps=8000
nl=3
nh=3

for len in 15
do
for gt in 0
do
for child_len in 5
do
python main.py --graph_type $gt\
               --graph_len $len\
               --max_child_chain_len $child_len\
               --n_layers $nl\
                --n_heads $nh\
                --if_train n\
                --train_epoch $train_epoch \
                --save_steps $save_steps\
                --if_test n\
                --if_plot n \
                --if_probe y\

done
done
done