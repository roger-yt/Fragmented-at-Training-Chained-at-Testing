export CUDA_VISIBLE_DEVICES="2"

train_epoch=14
save_steps=8000
nl=3
nh=3

for len in 10
do
for gt in 0 1 2 3 4
do
for child_len in 4 3 2
do
python main.py --graph_type $gt\
               --graph_len $len\
               --max_child_chain_len $child_len\
               --n_layers $nl\
                --n_heads $nh\
                --if_train y\
                --train_epoch $train_epoch \
                --save_steps $save_steps\
                --if_test y\
                --if_plot n \
                --if_probe n\

done
done
done