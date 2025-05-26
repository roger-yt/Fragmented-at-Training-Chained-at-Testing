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
import hydra
from omegaconf import DictConfig, OmegaConf
import re

_DCPATTERN = re.compile(r'^depth(\d+)_maxchild(\d+)$')
def parse_depth_maxchild(s: str):
    m = _DCPATTERN.match(s)
    if not m:
        return None
    depth, maxchild = m.groups()
    return int(depth), int(maxchild)
_TYPEPATTERN = re.compile(r'^type(\d+)$')
def parse_type(s: str):
    m = _TYPEPATTERN.match(s)
    if not m:
        return None
    typi = m.groups()
    return int(typi[0])

@hydra.main(config_path="configs", config_name="config_normal")
def main(cfg: DictConfig):
        tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer.model_name)
        tokenizer.pad_token = tokenizer.eos_token
        # Test_max_examples = Args.max_examples
        context_len = 2048  #len(Train_ds["input_ids"][0])
        model_cfg = MyGPT2Config(
                vocab_size=len(tokenizer),
                n_ctx=context_len,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                n_layer=cfg.model.n_layers,
                n_head=cfg.model.n_heads,
                max_position_embeddings=context_len,
                hidden_size=cfg.model.hidden_size
        )
        width = 2
        if cfg.draw.type == "standard":
                acc_types = ["whole",  "final", "te_ver", "te_val"]
        elif cfg.draw.type == "blur":
                acc_types = ["whole",  "final", "te_ver",  "te_val"]
        elif cfg.draw.type == "skip":
                acc_types = ["whole",  "final", "te_ver", "te_val"]
        name_type_map={"whole":"Whole acc", "final":"Final acc","te_val":"Values acc", "te_ver":"Vertices acc", "te_ver_comp": "Vertices acc full"}
        dc_map = {}
        for dir in os.listdir(cfg.draw.parent_dir):
                d, c = parse_depth_maxchild(dir)
                if d in dc_map.keys():
                        dc_map[d].append(c)
                else:
                        dc_map[d] = [c]
        dc_map = {k: sorted(v) for k, v in sorted(dc_map.items())}
        print("dc_map=", dc_map)

        all_child_chain_len = []
        for key in dc_map.keys():
                all_child_chain_len += dc_map[key]
        all_child_chain_len = sorted(list(set(all_child_chain_len)))

        color_ls = ["black", "red", "green", "blue", "purple", "brown","orange" , "gray", "pink", "cyan"]

        line_proxies = []
        labels = []
        if cfg.draw.mode == "main":
                fig, axes = plt.subplots(len(acc_types), len(dc_map), figsize=(6*len(dc_map), 15))
                print("axes:", axes)
                for o, acc_tp in enumerate(acc_types):
                        for j, leng in enumerate(dc_map.keys()):
                                if len(dc_map) == 1:
                                        axes_obj = axes[o]
                                elif len(dc_map) > 1:
                                        axes_obj = axes[o][j]
                                for k,child_chain_len in enumerate(dc_map[leng][::-1]):
                                        gs = [1]+[width]*(leng-1)
                                        Test_len = len(gs)
                                        data_dir = f"{cfg.draw.parent_dir}/depth{leng}_maxchild{child_chain_len}"
                                        Device = "cuda" if torch.cuda.is_available() else "cpu"
                                        print("Device:", Device)
                                        curves = []
                                        # for typi in range(5):
                                        for type_dir in os.listdir(data_dir):
                                                model_dir = f"{data_dir}/{type_dir}/outs_{cfg.model.name}"
                                                print("model_dir:", model_dir)
                                                if not os.path.exists(model_dir):
                                                        continue
                                                typi = parse_type(type_dir)
                                                print(typi, type_dir)
                                                gg = Goal_graph(
                                                        graph_shape=gs,
                                                        graph_type=typi,
                                                        context_lower=cfg.data.context_lower,
                                                        context_upper=cfg.data.context_upper,
                                                        context_div=cfg.data.context_div,
                                                        vocab_size=cfg.model.vocab_size,
                                                        env_val_num_low=cfg.data.env_val_num_low,
                                                        chain_val_num=cfg.data.chain_val_num,
                                                        leak_prob_node=cfg.data.leak_prob_node,
                                                        leak_prob_val=cfg.data.leak_prob_val,
                                                        addlen=cfg.data.addlen,
                                                        nearlen=cfg.data.nearlen,
                                                        tl_low=cfg.data.tl_low,
                                                        tokenizer=tokenizer
                                                )
                                                outs_path = f"{model_dir}/layer{cfg.model.n_layers}_head{cfg.model.n_heads}_hidden{cfg.model.hidden_size}"
                                                print("outs_path:", outs_path)
                                                map_str = cfg.draw.name
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
                                                                raise FileNotFoundError(f"No checkpoint directories found in {outs_path}. Please delete this directory or complete the training of this directory.")
                                                        print("model_path:", model_path)
                                                        Model = MyGPT2LMHeadModel.from_pretrained(model_path, config=model_cfg).to(Device)
                                                        if cfg.draw.type == "standard":
                                                                acc_map = do_test(gg, Model, tokenizer, cfg.data.max_examples, Test_len)
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

                                        axes_obj.plot(range(cfg.data.max_examples), mean_curve, label=f"child chain len={str(child_chain_len)}", color=color_ls[child_chain_len-2], linestyle='-', linewidth=2)
                                        axes_obj.fill_between(range(cfg.data.max_examples), mean_curve - std_curve, mean_curve + std_curve, color=color_ls[child_chain_len-2], alpha=0.2)
                                        # axes[o][j].plot(range(Args.max_examples), acc_map[acc_tp], label=f"child chain len={str(child_chain_len)}", color=color_ls[child_chain_len-2], linestyle='-', linewidth=2)
                                if o == len(acc_types)-1:
                                        axes_obj.set_xlabel('Shots Num', fontsize=24, fontweight='bold')
                                if j == 0:
                                        axes_obj.set_ylabel(f'{name_type_map[acc_tp]}', fontsize=24, fontweight='bold')
                                if o == 0:
                                        axes_obj.set_title(f"Depth={leng}", fontsize=24, fontweight='bold')
                                axes_obj.set_ylim(-0.05, 1.05)  # Ensure the range covers 0.0 to 1.0
                                axes_obj.set_yticks([0.0, 0.5, 1.0])  # Explicit tick marks at 0.0, 0.5, and 1.0

                                axes_obj.set_facecolor('lightgrey')
                                axes_obj.grid(True, which='both', color='white', linestyle='-', linewidth=0.7)
                                axes_obj.minorticks_on()

                                axes_obj.xaxis.set_major_locator(MultipleLocator(1))  # Major ticks every 2 units
                                axes_obj.yaxis.set_major_locator(MultipleLocator(0.5))  # Major ticks every 0.5 units
                                axes_obj.xaxis.set_minor_locator(MultipleLocator(1))  # Minor ticks every 1 unit
                                axes_obj.yaxis.set_minor_locator(MultipleLocator(0.5))  # Minor ticks every 0.25 units
                                axes_obj.tick_params(axis='x', colors='black', direction='in', length=6, width=2, labelsize=20)
                                axes_obj.tick_params(axis='y', colors='black', direction='in', length=6, width=2, labelsize=20)

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
                plt.savefig(f'fs_and_chain_len_{cfg.draw.model_size}.png', facecolor=fig.get_facecolor())


        if cfg.draw.mode == "ratio":
                fig, axes = plt.subplots(len(acc_types), 1, figsize=(8, 10))
                print("axes:", axes)
                ratio_map = {}
                for o, acc_tp in enumerate(acc_types):
                        for j, leng in enumerate(dc_map.keys()):
                                for k,child_chain_len in enumerate(dc_map[leng][::-1]):
                                        gs = [1]+[width]*(leng-1)
                                        Test_len = len(gs)
                                        data_dir = f"{cfg.draw.parent_dir}/depth{leng}_maxchild{child_chain_len}"
                                        Device = "cuda" if torch.cuda.is_available() else "cpu"
                                        print("Device:", Device)
                                        curves = []
                                        # for typi in range(5):
                                        max_accs = []
                                        for type_dir in os.listdir(data_dir):
                                                model_dir = f"{data_dir}/{type_dir}/outs_{cfg.model.name}"
                                                print("model_dir:", model_dir)
                                                if not os.path.exists(model_dir):
                                                        continue
                                                typi = parse_type(type_dir)
                                                print(typi, type_dir)
                                                gg = Goal_graph(
                                                        graph_shape=gs,
                                                        graph_type=typi,
                                                        context_lower=cfg.data.context_lower,
                                                        context_upper=cfg.data.context_upper,
                                                        context_div=cfg.data.context_div,
                                                        vocab_size=cfg.model.vocab_size,
                                                        env_val_num_low=cfg.data.env_val_num_low,
                                                        chain_val_num=cfg.data.chain_val_num,
                                                        leak_prob_node=cfg.data.leak_prob_node,
                                                        leak_prob_val=cfg.data.leak_prob_val,
                                                        addlen=cfg.data.addlen,
                                                        nearlen=cfg.data.nearlen,
                                                        tl_low=cfg.data.tl_low,
                                                        tokenizer=tokenizer
                                                )
                                                outs_path = f"{model_dir}/layer{cfg.model.n_layers}_head{cfg.model.n_heads}_hidden{cfg.model.hidden_size}"
                                                print("outs_path:", outs_path)
                                                map_str = cfg.draw.name
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
                                                                raise FileNotFoundError(f"No checkpoint directories found in {outs_path}. Please delete this directory or complete the training of this directory.")
                                                        print("model_path:", model_path)
                                                        Model = MyGPT2LMHeadModel.from_pretrained(model_path, config=model_cfg).to(Device)
                                                        if cfg.draw.type == "standard":
                                                                acc_map = do_test(gg, Model, tokenizer, cfg.data.max_examples, Test_len)
                                                        # acc_map = do_test(My_Goal_graph, Model, Tokenizer, Test_max_examples, Test_len)
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
                plt.savefig(f'ratio_{cfg.draw.model_size}.png', facecolor=fig.get_facecolor())

if __name__ == "__main__":
    main()