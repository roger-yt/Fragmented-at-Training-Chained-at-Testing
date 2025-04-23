export CUDA_VISIBLE_DEVICES="0"


train_epoch=5
save_steps=8000

for nl in 1 2
do
for nh in 1 2
do
for len in 5
do
for gt in 0 # 5
do
for child_len in 3 
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
done
done

for nl in 3
do
for nh in 3
do
for len in 5
do
for gt in 0 # 5
do
for child_len in 3 
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
done
done