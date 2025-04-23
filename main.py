import os
from transformers import AutoTokenizer, Trainer, TrainingArguments
from utils.collators import My_collator
from utils.configs import MyGPT2Config
import torch
import numpy as np

import argparse

from utils.networks import MyGPT2LMHeadModel
from data_structure_related.data_structure import Goal_graph
from utils.utils import *
import logging
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="gpt2")
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


parser.add_argument("--n_layers", type=int, default=1)
parser.add_argument("--n_heads", type=int, default=1) 
parser.add_argument("--hidden_size", type=int, default=720)

parser.add_argument("--if_train", type=str, default="y")
parser.add_argument("--if_upload", type=str, default="y")
parser.add_argument("--train_epoch", type=int, default=10)
parser.add_argument("--per_device_train_batch_size", type=int, default=8)
parser.add_argument("--per_device_eval_batch_size", type=int, default=8)
parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
parser.add_argument("--lr", type=float, default=5e-05)
parser.add_argument("--save_steps", type=int, default=1000)
parser.add_argument("--if_test", type=str, default="y")
parser.add_argument("--if_plot", type=str, default="y")
parser.add_argument("--if_probe", type=str, default="y")
parser.add_argument("--probe_mean_num", type=int, default=10)
parser.add_argument("--test_epoch", type=int, default=-1)
parser.add_argument("--if_in_colab", type=str, default="n")

Args = parser.parse_args()

print("Args:", Args)

assert Args.graph_len > Args.max_child_chain_len

if Args.if_in_colab=="y":
        os.chdir("/content/drive/MyDrive/ICL/CoT_handin")
        print("Current working directory:", os.getcwd())
Graph_shape = [Args.graph_width]*Args.merge_pos + [1] + [Args.graph_width]*(Args.graph_len-Args.merge_pos-1)
Test_len = len(Graph_shape)
Test_max_examples = Args.max_examples
Tokenizer = AutoTokenizer.from_pretrained(Args.model)
Tokenizer.pad_token = Tokenizer.eos_token
Token_num = len(Tokenizer)
print("Token_num:", Token_num)

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

type_dir = f"{data_dir}/{foot_str}/type{str(Args.graph_type)}"
if not os.path.exists(type_dir):
        os.makedirs(type_dir)
model_dir = f"{type_dir}/outs_{Args.model}"

Train_ds, Valid_ds = prepare_training_data(My_Goal_graph, Args, Tokenizer, data_dir, type_dir)
Context_len = 2048  

Config = MyGPT2Config(
    vocab_size=len(Tokenizer),
    n_ctx=Context_len,
    bos_token_id=Tokenizer.bos_token_id,
    eos_token_id=Tokenizer.eos_token_id,
    n_layer=Args.n_layers,
    n_head=Args.n_heads,
    max_position_embeddings=Context_len,
        hidden_size=Args.hidden_size,
)

Device = "cuda" if torch.cuda.is_available() else "cpu"
Model = MyGPT2LMHeadModel(Config).to(Device)
print("Model:", Model._get_name())
print("Device:", Device)
num_params = sum(p.numel() for p in Model.parameters())
print("num_params:", num_params)
Data_collator = My_collator(Tokenizer)


print("model_dir:", model_dir)
if not os.path.exists(model_dir):
        print("Create model_dir:", model_dir)
        os.makedirs(model_dir)

outs_path = f"{model_dir}/layer{Args.n_layers}_head{Args.n_heads}_hidden{Args.hidden_size}"

print("outs_path:", outs_path)
Train_Args = TrainingArguments(
    output_dir=outs_path,
    eval_strategy="epoch",
    num_train_epochs=Args.train_epoch,
    save_steps=Args.save_steps,
        per_device_eval_batch_size=Args.per_device_eval_batch_size,                                                                                                               
        per_device_train_batch_size=Args.per_device_train_batch_size,  
        report_to="none",
        gradient_accumulation_steps=Args.gradient_accumulation_steps,
        learning_rate=Args.lr,
)

def get_latest_checkpoint(outs_path):
        checkpoints = [d for d in os.listdir(outs_path) if d.startswith("checkpoint-")]
        if len(checkpoints) == 0:
            return None, None
        latest_checkpoint = max(checkpoints, key=lambda x: int(x.split("-")[1]))
        return os.path.join(outs_path, latest_checkpoint), latest_checkpoint

if Args.if_train=="y":
        print("training")
        trainer = Trainer(
        model=Model,
        tokenizer=Tokenizer,
        args=Train_Args,
        data_collator=Data_collator,
        train_dataset=Train_ds,
        eval_dataset=Valid_ds
        )
        latest_checkpoint, ckpt = get_latest_checkpoint(outs_path)
        if latest_checkpoint is not None:
                trainer.train(resume_from_checkpoint=True)
        else:
                trainer.train()


if Args.if_test=="y" or Args.if_probe=="y" or Args.if_plot=="y":
        if Args.test_epoch==-1:
               checkpoint_dirs = [d for d in os.listdir(outs_path) if d.startswith("checkpoint-")]
               if checkpoint_dirs:
                        latest_checkpoint = max(checkpoint_dirs, key=lambda x: int(x.split("-")[1]))
                        model_path = os.path.join(outs_path, latest_checkpoint)
                        test_epoch = int(latest_checkpoint.split("-")[1])
               else:
                        raise FileNotFoundError(f"No checkpoint directories found in {outs_path}")
        else:
                model_path = f"{outs_path}/checkpoint-{Args.test_epoch}"
                test_epoch = Args.test_epoch
        Model = MyGPT2LMHeadModel.from_pretrained(model_path, config=Config).to(Device)
        training_args = torch.load(f"{model_path}/training_args.bin")
        print("training_args=", training_args)


if Args.if_test=="y":
        trainer = Trainer(
                model=Model,
                tokenizer=Tokenizer,
                args=Train_Args,
                data_collator=Data_collator,
                train_dataset=Train_ds,
                eval_dataset=Valid_ds
        )
        log_path = f"{outs_path}/test_epoch{test_epoch}_len{Test_len}.log"
        print("log_path:", log_path)
        handler = logging.FileHandler(log_path, mode='w')
        handler.setFormatter(logging.Formatter('%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s'))
        logger = logging.getLogger('testing')
        logger.setLevel(logging.DEBUG)
        logger.addHandler(handler)
        logger.addHandler(logging.StreamHandler(sys.stdout))
        logger.info(trainer.evaluate())
        do_test(My_Goal_graph, Model, Tokenizer, Test_max_examples, Test_len, logger)

        
if Args.if_probe =="y":
        log_path = f"{outs_path}/prob_epoch{Args.test_epoch}_meannum{Args.probe_mean_num}.log"
        handler = logging.FileHandler(log_path, mode='w')
        handler.setFormatter(logging.Formatter('%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s'))
        logger = logging.getLogger('probing')
        logger.setLevel(logging.DEBUG)
        logger.addHandler(handler)
        logger.addHandler(logging.StreamHandler(sys.stdout))
        logger.info(f"path len={Test_len}")

        logger.info("mk probing")
        do_probe(My_Goal_graph, Model, Tokenizer, 5, Args.max_child_chain_len, Test_len, 10, logger, Device, "mk", "val")
        
        logger.info("test probing")
        do_probe(My_Goal_graph, Model, Tokenizer, 5, Args.max_child_chain_len, Test_len, 10, logger, Device, "test", "val")

if Args.if_plot=="y":
        do_plot(Args, My_Goal_graph, Model, Tokenizer, 2, Test_len,  Device, Train_ds, outs_path, Args.test_epoch)