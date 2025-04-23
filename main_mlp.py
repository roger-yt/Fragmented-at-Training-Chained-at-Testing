import os
from transformers import AutoTokenizer
from utils.configs import MyMLPConfig
import pickle as pkl
import torch
from torch.utils.data.dataloader import DataLoader
import argparse


from utils.networks import MyMLP
from data_structure_related.data_structure import Goal_graph
import logging
import sys
from utils.trainers import MLPTrainer
from utils.utils import *

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="mlp")
parser.add_argument("--num_icl_train_traces", type=int, default=5000)
parser.add_argument("--num_icl_valid_traces", type=int, default=100)
parser.add_argument("--num_mk_train_traces", type=int, default=20000)
parser.add_argument("--num_mk_valid_traces", type=int, default=100)
parser.add_argument("--max_examples", type=int, default=5)

parser.add_argument("--graph_type", type=int)
parser.add_argument("--graph_len", type=int)
parser.add_argument("--graph_width", type=int, default=2)
parser.add_argument("--merge_pos", type=int, default=0)

parser.add_argument("--max_child_chain_len", type=int, default=2)
parser.add_argument("--vocab_size", type=int, default=52)
parser.add_argument("--env_val_num_low", type=int, default=10)
parser.add_argument("--chain_val_num", type=int, default=50)
parser.add_argument("--leak_prob_node", type=float, default=0.0)
parser.add_argument("--leak_prob_val", type=float, default=0.0)
parser.add_argument("--tl_low", type=int, default=4)
parser.add_argument("--addlen", type=int, default=5)
parser.add_argument("--nearlen", type=int, default=100)
parser.add_argument("--context_lower", type=int, default=1)
parser.add_argument("--context_upper", type=int, default=7)
parser.add_argument("--context_div", type=int, default=7)

parser.add_argument("--n_layers", type=int, default=2)
parser.add_argument("--hidden_size", type=int, default=1000)
parser.add_argument("--window_size", type=int, default=100)


parser.add_argument("--if_train", type=str, default="y")
parser.add_argument("--train_epoch", type=int, default=10)
parser.add_argument("--save_steps", type=int, default=1000)
parser.add_argument("--if_test", type=str, default="y")
parser.add_argument("--if_plot", type=str, default="y")
parser.add_argument("--if_probe", type=str, default="y")
parser.add_argument("--probe_mean_num", type=int, default=10)
parser.add_argument("--test_epoch", type=int, default=-1)
parser.add_argument("--if_in_colab", type=str, default="n")

Args = parser.parse_args()

print("Args:", Args)

if Args.if_in_colab=="y":
        os.chdir("/content/drive/MyDrive/ICL/CoT_theory_mask")
        print("Current working directory:", os.getcwd())

Graph_shape = [1]+[Args.graph_width]*(Args.graph_len-1)
Test_len = len(Graph_shape)
Test_max_examples = Args.max_examples

Tokenizer = AutoTokenizer.from_pretrained("gpt2")
Tokenizer.pad_token = Tokenizer.eos_token
Token_num = len(Tokenizer)
My_Goal_graph = Goal_graph(graph_shape=Graph_shape,
                           graph_type=Args.graph_type,
                        context_lower=Args.context_lower,
                        context_upper=Args.context_upper,
                        context_div=Args.context_div,
                           vocab_size=Args.vocab_size,
                          env_val_num_low=Args.env_val_num_low,
                          chain_val_num=Args.chain_val_num,
                                leak_prob_node=Args.leak_prob_node,
                                leak_prob_val=Args.leak_prob_val,
                           addlen=Args.addlen,
                           nearlen=Args.nearlen,
                           tl_low=Args.tl_low,
                           tokenizer=Tokenizer
                           )
data_dir = f"data_and_models"
shape_dir = f"len{Args.graph_len}_width{Args.graph_width}_merge{Args.merge_pos}"
if not os.path.exists(f"{data_dir}/{shape_dir}"):
        os.makedirs(f"{data_dir}/{shape_dir}")

foot_str = f"{shape_dir}/maxchildlen{str(Args.max_child_chain_len)}\
_cl{Args.context_lower}_cu{Args.context_upper}_cd{Args.context_div}_vocab{str(Args.vocab_size)}_envaln{str(Args.env_val_num_low)}\
_chainvaln{str(Args.chain_val_num)}_lkpn{Args.leak_prob_node}_lkpv{Args.leak_prob_val}\
_addlen{Args.addlen}_nearlen{Args.nearlen}_tl{Args.tl_low}_shot{Args.max_examples}_icl{str(Args.num_icl_train_traces)}_mk{str(Args.num_mk_train_traces)}"

Alal_token_ids = My_Goal_graph.alal_token_ids
print("len(All_token_ids):", len(Alal_token_ids))
# print("All_token_ids:", All_token_ids)

type_dir = f"{data_dir}/{foot_str}/type{str(Args.graph_type)}"
if not os.path.exists(type_dir):
        os.makedirs(type_dir)
model_dir = f"{type_dir}/outs_{Args.model}"
if not os.path.exists(model_dir):
        os.makedirs(model_dir)

Train_ds, Valid_ds = prepare_training_data(My_Goal_graph, Args, Tokenizer, data_dir, type_dir)
Context_len = len(Train_ds["input_ids"][0])

mlp_data_path = f"{model_dir}/data_window{Args.window_size}.pkl"
if not os.path.exists(mlp_data_path):
        Train_ds = get_mlp_dataset(Args.window_size, Train_ds, Tokenizer)
        Valid_ds = get_mlp_dataset(Args.window_size, Valid_ds, Tokenizer)
        pkl.dump((Train_ds, Valid_ds), open(mlp_data_path, "wb"))
Train_ds, Valid_ds = pkl.load(open(mlp_data_path, "rb"))

for i in range(-100, -95):
        print(len(Train_ds[i]["input"]))
        # print(Train_ds[i]["input"])
        print(Tokenizer.decode(Train_ds[i]["input"].int()))
        print()

Device = "cuda" if torch.cuda.is_available() else "cpu"

Config = MyMLPConfig(window_size = Args.window_size,
        in_dim=Args.window_size * len(Alal_token_ids), 
              out_dim=len(Tokenizer), 
              hidden_dim=Args.hidden_size, 
              layer_num=Args.n_layers, 
              all_token_ids=Alal_token_ids, 
              vocab_size=len(Tokenizer))
print("Config:", Config)
Model = MyMLP(Config).to(Device)

def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

print("Model parameter number:", get_parameter_number(Model))


print("Model:", Model._get_name())
print("Device:", Device)

if not os.path.exists(model_dir):
        os.makedirs(model_dir)

outs_path = f"{model_dir}/layer{Args.n_layers}_hidden{Args.hidden_size}_window{Args.window_size}"

train_dataloader = DataLoader(Train_ds, batch_size=10000, shuffle=True)
eval_dataloader = DataLoader(Valid_ds, batch_size=10000, shuffle=True)
if Args.if_train=="y":
        print("training")
        # Model = nn.DataParallel(Model)
        optimizer = torch.optim.Adam(Model.parameters(), lr=5e-5, weight_decay=0.0001)
        trainer = MLPTrainer(Model, optimizer)
        if (outs_path is not None) and (not os.path.exists(outs_path)):
            os.makedirs(outs_path)
        trainer.train(train_dataloader, eval_dataloader, Args.train_epoch, outs_path)

if Args.if_test=="y":
        if Args.test_epoch==-1:
               checkpoint_dirs = [d for d in os.listdir(outs_path) if d.startswith("checkpoint-")]
               if checkpoint_dirs:
                        latest_checkpoint = max(checkpoint_dirs, key=lambda x: int(x.split("-")[1].split(".")[0]))
                        max_epoch = int(latest_checkpoint.split("-")[1].split(".")[0])
                        model_path = os.path.join(outs_path, latest_checkpoint)
                        log_path = f"{outs_path}/test_epoch{max_epoch}.log"
               else:
                        raise FileNotFoundError(f"No checkpoint directories found in {outs_path}")
        else:
                model_path = f"{outs_path}/checkpoint-{Args.test_epoch}.pt"
                log_path = f"{outs_path}/test_epoch{Args.test_epoch}.log"
        Model.load_state_dict(torch.load(model_path)["model_state_dict"])
        optimizer = torch.optim.Adam(Model.parameters(), lr=0.001, weight_decay=0.0001)
        trainer = MLPTrainer(Model, optimizer)
        logging.basicConfig(filename=log_path,
                filemode='w',
                format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                datefmt='%H:%M:%S',
                level=logging.DEBUG)
        logging.info("Begin testing")
        logger = logging.getLogger('testing')
        logger.addHandler(logging.StreamHandler(sys.stdout))
        eval_acc, eval_loss = trainer.evaluate(eval_dataloader)
        logger.info(f"Eval Loss: {eval_loss}")
        do_test(My_Goal_graph, Model, Tokenizer, Test_max_examples, Test_len, logger)