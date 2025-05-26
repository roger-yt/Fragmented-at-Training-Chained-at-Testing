export CUDA_VISIBLE_DEVICES="0"
len=5
gt=0
child_len=3

data_dir="data_and_models/quick_start/depth${len}_maxchild${child_len}/type${gt}"
if [ ! -d "$data_dir" ]; then
  mkdir -p "$data_dir"          # -p also builds parents
  echo "Created $data_dir"
fi

python data_gen.py \
    --config-name config_normal.yaml\
    graph.len=$len \
    graph.type=$gt \
    data.max_child_chain_len=$child_len\
    paths.data_dir=$data_dir

python main.py \
     --config-name config_normal.yaml\
    graph.len=$len \
    graph.type=$gt \
    data.max_child_chain_len=$child_len\
    paths.data_dir=$data_dir\
    modes.train=true\
    modes.test=true\

parent_dir="data_and_models/quick_start"

python draw.py \
     --config-name config_normal.yaml\
    draw.parent_dir=$parent_dir\
    draw.mode=main\
    draw.model_size=quick_start

