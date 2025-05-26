export CUDA_VISIBLE_DEVICES="1"
for len in 5
do
for gt in 0 1 2 3 4
do
for child_len in 3 2
do

data_dir="data_and_models/gptsmall/depth${len}_maxchild${child_len}/type${gt}"
if [ ! -d "$data_dir" ]; then
  mkdir -p "$data_dir"          # -p also builds parents
  echo "Created $data_dir"
fi
python data_gen.py \
    --config-name config_gptsmall.yaml\
    graph.len=$len \
    graph.type=$gt \
    data.max_child_chain_len=$child_len\
    paths.data_dir=$data_dir

python main.py \
     --config-name config_gptsmall.yaml\
    graph.len=$len \
    graph.type=$gt \
    data.max_child_chain_len=$child_len\
    paths.data_dir=$data_dir\

done
done
done