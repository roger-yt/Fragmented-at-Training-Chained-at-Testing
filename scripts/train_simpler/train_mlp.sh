export CUDA_VISIBLE_DEVICES="0"
hidden_size=1000
len=5
child_len=3
wind=200
for nl in  2 3 4
do
for gt in 0
do
python main_mlp.py --graph_type $gt\
               --graph_len $len\
               --max_child_chain_len $child_len\
               --n_layers $nl\
                --hidden_size $hidden_size\
                --window_size $wind\
                --if_train y\
                --train_epoch 20 \
                --save_steps 2780\
                --if_test y\
                --if_plot n \
                --if_probe n 

done
done