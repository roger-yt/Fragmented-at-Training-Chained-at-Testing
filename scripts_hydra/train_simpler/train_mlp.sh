export CUDA_VISIBLE_DEVICES="0"
for len in 5
do
for gt in 0
do
for child_len in 3
do
for nl in 2 3 4
do

data_dir="data_and_models/normal_and_simpler/depth${len}_maxchild${child_len}/type${gt}"
if [ ! -d "$data_dir" ]; then
  mkdir -p "$data_dir"          # -p also builds parents
  echo "Created $data_dir"
fi
python data_gen.py \
    --config-name config_mlp.yaml\
    graph.len=$len \
    graph.type=$gt \
    data.max_child_chain_len=$child_len\
    paths.data_dir=$data_dir

python main_mlp_hydra.py \
     --config-name config_mlp.yaml\
    graph.len=$len \
    graph.type=$gt \
    data.max_child_chain_len=$child_len\
    paths.data_dir=$data_dir\
    model.n_layers=$nl

done
done
done
done