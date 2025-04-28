import os
from transformers import AutoTokenizer
from utils.configs import MyGPT2Config
import pickle as pkl
import torch
import argparse

from utils.networks import MyGPT2LMHeadModel
from data_structure_related.data_structure import Goal_graph
from utils.utils import *
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator

parser = argparse.ArgumentParser()

parser.add_argument("--model", type=str, default="gpt2")
parser.add_argument("--mode", type=str, default="main")
parser.add_argument("--retain_ratio", type=float, default=0.2)
parser.add_argument("--name", type=str)

parser.add_argument("--type", type=str, default="standard")


parser.add_argument("--num_icl_train_traces", type=int, default=5000)
parser.add_argument("--num_icl_valid_traces", type=int, default=100)
parser.add_argument("--num_mk_train_traces", type=int, default=20000)
parser.add_argument("--num_mk_valid_traces", type=int, default=100)
parser.add_argument("--max_examples", type=int, default=5)

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
parser.add_argument("--hidden_size", type=int, default=100)
parser.add_argument("--model_size", type=str)


Args = parser.parse_args()

print("Args:", Args)

Tokenizer = AutoTokenizer.from_pretrained(Args.model)
Tokenizer.pad_token = Tokenizer.eos_token
Token_num = len(Tokenizer)
Test_max_examples = Args.max_examples
Context_len = 2048  #len(Train_ds["input_ids"][0])
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

width = 2
if Args.type == "standard":
        acc_types = ["whole",  "final", "te_ver", "te_val"]
elif Args.type == "blur":
        acc_types = ["whole",  "final", "te_ver",  "te_val"]
elif Args.type == "skip":
        acc_types = ["whole",  "final", "te_ver", "te_val"]
name_type_map={"whole":"Whole acc", "final":"Final acc","te_val":"Values acc", "te_ver":"Vertices acc", "te_ver_comp": "Vertices acc full"}

if Args.n_layers == 3:
        width_map = {
                2:{5: [2, 3], 8:[2, 3, 4], 10: [2, 3, 4], 13:[2, 3, 4, 5], 15: [2, 3, 4, 6]}
        }

if Args.n_layers == 12:
        width_map = {
                        2:{5: [2, 3], 8:[2, 3, 4], 10: [2, 3, 4], 13:[2, 3, 4, 5], 15: [2, 3, 4, 6]}
                        }
elif Args.n_layers == 36:
        width_map = {
                        2:{5: [2, 3], 8:[2, 3, 4, 5], 10: [2, 3, 4, 5], 13:[2, 3, 4, 5, 6], 15: [2, 3, 4, 6, 7]}
                        }

if Args.mode == "main":
        all_child_chain_len = []
        for key in width_map.keys():
                for key2 in width_map[key].keys():
                        all_child_chain_len += width_map[key][key2]
        all_child_chain_len = sorted(list(set(all_child_chain_len)))

        color_ls = ["black", "red", "green", "blue", "purple", "brown","orange" , "gray", "pink", "cyan"]

        out_maps = []
        draw_dir = "draws"
        if not os.path.exists(draw_dir):
                os.makedirs(draw_dir)

        line_proxies = []
        labels = []
        fig, axes = plt.subplots(len(acc_types), len(width_map[width]), figsize=(30, 15))
        print("axes:", axes)
        for o, acc_tp in enumerate(acc_types):
                for j, leng in enumerate(width_map[width].keys()):
                        for k,child_chain_len in enumerate(width_map[width][leng][::-1]):
                                Graph_shape = [1]+[width]*(leng-1)
                                Test_len = len(Graph_shape)
                                data_dir = f"data_and_models"
                                shape_dir = f"len{leng}_width{width}_merge{0}"
                                foot_str = f"{shape_dir}/maxchildlen{str(child_chain_len)}\
_cl{Args.context_lower}_cu{Args.context_upper}_cd{Args.context_div}_vocab{str(Args.vocab_size)}_envaln{str(Args.env_val_num_low)}\
_chainvaln{str(Args.chain_val_num)}_lkpn{Args.leak_prob_node}_lkpv{Args.leak_prob_val}\
_addlen{Args.addlen}_nearlen{Args.nearlen}_tl{Args.tl_low}_shot{Args.max_examples}_icl{str(Args.num_icl_train_traces)}_mk{str(Args.num_mk_train_traces)}"
                                data_dir = f"data_and_models"
                                Device = "cuda" if torch.cuda.is_available() else "cpu"
                                print("Device:", Device)
                                curves = []
                                for typi in range(5):
                                        model_dir = f"{data_dir}/{foot_str}/type{typi}/outs_{Args.model}"
                                        print("model_dir:", model_dir)
                                        if not os.path.exists(model_dir):
                                                continue
                                        My_Goal_graph = Goal_graph(graph_shape=Graph_shape,
                                                                   graph_type=typi,
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
                                        outs_path = f"{model_dir}/layer{Args.n_layers}_head{Args.n_heads}_hidden{Args.hidden_size}"
                                        print("outs_path:", outs_path)
                                        map_str = Args.name
                                        if Args.type == "blur":
                                                map_str += "_blur"
                                        elif Args.type == "skip":
                                                map_str += "_skip"+"_"+str(Args.retain_ratio)
                                        if os.path.exists(f"{outs_path}/{map_str}.pkl"):
                                                print("hi")
                                                with open(f"{outs_path}/{map_str}.pkl", "rb") as f:
                                                        acc_map = pkl.load(f)
                                        else:
                                                checkpoint_dirs = [d for d in os.listdir(outs_path) if d.startswith("checkpoint-")]
                                                if checkpoint_dirs:
                                                        latest_checkpoint = max(checkpoint_dirs, key=lambda x: int(x.split("-")[1]))
                                                        model_path = os.path.join(outs_path, latest_checkpoint)
                                                else:
                                                        raise FileNotFoundError(f"No checkpoint directories found in {outs_path}")
                                                print("model_path:", model_path)
                                                Model = MyGPT2LMHeadModel.from_pretrained(model_path, config=Config).to(Device)
                                                if Args.type == "standard":
                                                        acc_map = do_test(My_Goal_graph, Model, Tokenizer, Test_max_examples, Test_len)
                                                elif Args.type == "blur":
                                                        acc_map = do_test_blur(My_Goal_graph, Model, Tokenizer, Test_max_examples, Test_len)
                                                elif Args.type == "skip":
                                                        acc_map = do_test_skip(My_Goal_graph, Model, Tokenizer, Test_max_examples, Test_len, retain_ratio=Args.retain_ratio)
                                                # acc_map = do_test(My_Goal_graph, Model, Tokenizer, Test_max_examples, Test_len)
                                                pkl.dump(acc_map, open(f"{outs_path}/{map_str}.pkl", "wb"))
                                        print("acc_map:", acc_map)
                                        curves.append(acc_map[acc_tp])
                                curves = np.array(curves)
                                print("curves.shape:", curves.shape)
                                mean_curve = np.mean(curves, axis=0)
                                print("mean_curve.shape:", mean_curve.shape)
                                std_curve = np.std(curves, axis=0)
                                print("std_curve.shape:", std_curve.shape)

                                axes[o][j].plot(range(Args.max_examples), mean_curve, label=f"child chain len={str(child_chain_len)}", color=color_ls[child_chain_len-2], linestyle='-', linewidth=2)
                                axes[o][j].fill_between(range(Args.max_examples), mean_curve - std_curve, mean_curve + std_curve, color=color_ls[child_chain_len-2], alpha=0.2)
                                # axes[o][j].plot(range(Args.max_examples), acc_map[acc_tp], label=f"child chain len={str(child_chain_len)}", color=color_ls[child_chain_len-2], linestyle='-', linewidth=2)
                        if o == len(acc_types)-1:
                                axes[o][j].set_xlabel('Shots Num', fontsize=24, fontweight='bold')
                        if j == 0:
                                axes[o][j].set_ylabel(f'{name_type_map[acc_tp]}', fontsize=24, fontweight='bold')
                        if o == 0:
                                axes[o][j].set_title(f"Depth={leng}", fontsize=24, fontweight='bold')
                        axes[o][j].set_ylim(-0.05, 1.05)  # Ensure the range covers 0.0 to 1.0
                        axes[o][j].set_yticks([0.0, 0.5, 1.0])  # Explicit tick marks at 0.0, 0.5, and 1.0

                        axes[o][j].set_facecolor('lightgrey')
                        axes[o][j].grid(True, which='both', color='white', linestyle='-', linewidth=0.7)
                        axes[o][j].minorticks_on()

                        axes[o][j].xaxis.set_major_locator(MultipleLocator(1))  # Major ticks every 2 units
                        axes[o][j].yaxis.set_major_locator(MultipleLocator(0.5))  # Major ticks every 0.5 units
                        axes[o][j].xaxis.set_minor_locator(MultipleLocator(1))  # Minor ticks every 1 unit
                        axes[o][j].yaxis.set_minor_locator(MultipleLocator(0.5))  # Minor ticks every 0.25 units
                        axes[o][j].tick_params(axis='x', colors='black', direction='in', length=6, width=2, labelsize=20)
                        axes[o][j].tick_params(axis='y', colors='black', direction='in', length=6, width=2, labelsize=20)

                        # Show legend
                        # axes[o][j].legend(frameon=True, loc='upper left',bbox_to_anchor=(1.0, 1.0), fontsize=20)
        for child_chain_len in all_child_chain_len:
                line_proxies.append(plt.Line2D([0], [0], color=color_ls[child_chain_len-2], linewidth=2))
                labels.append(f"child_chain_len={str(child_chain_len)}")
        fig.legend(handles=line_proxies, labels=labels, loc='upper center', bbox_to_anchor=(0.49, 1.0), ncol=len(all_child_chain_len), fontsize=24)

        # Adjust layout to prevent overlap
        plt.tight_layout(rect=[0, 0, 0.95, 0.95])
        
        # plt.show()
        # Save the figure with a grey background
        plt.savefig(f'fs_and_chain_len_{Args.model_size}.png', facecolor=fig.get_facecolor())


if Args.mode == "ratio":
        all_child_chain_len = []
        for key in width_map.keys():
                for key2 in width_map[key].keys():
                        all_child_chain_len += width_map[key][key2]
        all_child_chain_len = sorted(list(set(all_child_chain_len)))


        color_ls = ["black", "red", "green", "blue", "purple", "cyan","orange" , "gray", "brown", "pink"]

        out_maps = []
        draw_dir = "draws"
        if not os.path.exists(draw_dir):
                os.makedirs(draw_dir)

        line_proxies = []
        labels = []
        fig, axes = plt.subplots(len(acc_types), 1, figsize=(8, 10))
        print("axes:", axes)
        ratio_map = {}
        for o, acc_tp in enumerate(acc_types):
                for j, leng in enumerate(width_map[width].keys()):
                        for k,child_chain_len in enumerate(width_map[width][leng][::-1]):
                                Graph_shape = [1]+[width]*(leng-1)
                                Test_len = len(Graph_shape)
                                data_dir = f"data_and_models"
                                shape_dir = f"len{leng}_width{width}_merge{0}"
                                foot_str = f"{shape_dir}/maxchildlen{str(child_chain_len)}\
_cl{Args.context_lower}_cu{Args.context_upper}_cd{Args.context_div}_vocab{str(Args.vocab_size)}_envaln{str(Args.env_val_num_low)}\
_chainvaln{str(Args.chain_val_num)}_lkpn{Args.leak_prob_node}_lkpv{Args.leak_prob_val}\
_addlen{Args.addlen}_nearlen{Args.nearlen}_tl{Args.tl_low}_shot{Args.max_examples}_icl{str(Args.num_icl_train_traces)}_mk{str(Args.num_mk_train_traces)}"
                                data_dir = f"data_and_models"
                                Device = "cuda" if torch.cuda.is_available() else "cpu"
                                print("Device:", Device)
                                max_accs = []
                                for typi in range(5):
                                        model_dir = f"{data_dir}/{foot_str}/type{typi}/outs_{Args.model}"
                                        if not os.path.exists(model_dir):
                                                continue
                                        My_Goal_graph = Goal_graph(graph_shape=Graph_shape,
                                                                   graph_type=typi,
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
                                        outs_path = f"{model_dir}/layer{Args.n_layers}_head{Args.n_heads}_hidden{Args.hidden_size}"
                                        print("outs_path:", outs_path)
                                        map_str = Args.name
                                        if Args.type == "blur":
                                                map_str += "_blur"
                                        elif Args.type == "skip":
                                                map_str += "_skip"+ "_"+str(Args.retain_ratio)
                                        if os.path.exists(f"{outs_path}/{map_str}.pkl"):
                                                with open(f"{outs_path}/{map_str}.pkl", "rb") as f:
                                                        acc_map = pkl.load(f)
                                        else:
                                                checkpoint_dirs = [d for d in os.listdir(outs_path) if d.startswith("checkpoint-")]
                                                if checkpoint_dirs:
                                                        latest_checkpoint = max(checkpoint_dirs, key=lambda x: int(x.split("-")[1]))
                                                        model_path = os.path.join(outs_path, latest_checkpoint)
                                                else:
                                                        raise FileNotFoundError(f"No checkpoint directories found in {outs_path}")
                                                print("model_path:", model_path)
                                                Model = MyGPT2LMHeadModel.from_pretrained(model_path, config=Config).to(Device)
                                                if Args.type == "standard":
                                                        acc_map = do_test(My_Goal_graph, Model, Tokenizer, Test_max_examples, Test_len)
                                                elif Args.type == "blur":
                                                        acc_map = do_test_blur(My_Goal_graph, Model, Tokenizer, Test_max_examples, Test_len)
                                                elif Args.type == "skip":
                                                        acc_map = do_test_skip(My_Goal_graph, Model, Tokenizer, Test_max_examples, Test_len, retain_ratio=Args.retain_ratio)
                                                pkl.dump(acc_map, open(f"{outs_path}/{map_str}.pkl", "wb"))
                                        maxi = np.max(acc_map[acc_tp])
                                        print(f"leng={leng}, child_chain_len={child_chain_len}, max acc={maxi}")
                                        max_accs.append(maxi)
                                ratio_map[child_chain_len/leng] = max_accs
                sorted_ratio_map = dict(sorted(ratio_map.items()))
                ratio_ls = []
                mean_ls = []
                std_ls = []
                for ratio, max_accs in sorted_ratio_map.items():
                        ratio_ls.append(ratio)
                        mean_ls.append(np.mean(max_accs))
                        std_ls.append(np.std(max_accs))
                print("ratio_ls:", ratio_ls)
                print("mean_ls:", mean_ls)
                print("std_ls:", std_ls)
                axes[o].plot(ratio_ls, mean_ls, label=f"depth={leng}", color="brown", linestyle='-', linewidth=2)
                axes[o].fill_between(ratio_ls, np.array(mean_ls) - np.array(std_ls), np.array(mean_ls) + np.array(std_ls), color="brown", alpha=0.2)
                if o == len(acc_types)-1:
                        axes[o].set_xlabel('Relative knowledge ratio', fontsize=24, fontweight='bold')
                axes[o].set_ylabel(f'{name_type_map[acc_tp]}', fontsize=24, fontweight='bold')
                axes[o].set_facecolor('lightgrey')
                axes[o].grid(True, which='both', color='white', linestyle='-', linewidth=0.7)
                axes[o].minorticks_on()
                axes[o].xaxis.set_major_locator(MultipleLocator(0.1))
                axes[o].yaxis.set_major_locator(MultipleLocator(0.5))
                axes[o].xaxis.set_minor_locator(MultipleLocator(0.05))
                axes[o].yaxis.set_minor_locator(MultipleLocator(0.25))
                axes[o].tick_params(axis='x', colors='black', direction='in', length=6, width=2, labelsize=20)
                axes[o].tick_params(axis='y', colors='black', direction='in', length=6, width=2, labelsize=20)
        # Adjust layout to prevent overlap
        plt.tight_layout(rect=[0, 0, 0.95, 0.95])

        # Save the figure with a grey background
        plt.savefig(f'ratio_{Args.model_size}.png', facecolor=fig.get_facecolor())