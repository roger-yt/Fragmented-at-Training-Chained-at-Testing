import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset, zoomed_inset_axes

from matplotlib.ticker import MultipleLocator
from matplotlib.patches import Rectangle

import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
import copy
from tqdm import tqdm
import os
import pickle as pkl

from datasets import Dataset, DatasetDict
from demonstration.probing_related import probing_MLP

from torch.utils.data.dataloader import DataLoader

import numpy as np
import random
from demonstration.text_attn import generate
import json

plt.rcParams.update({
    'font.size': 22,         # General font size
    'axes.titlesize': 30,    # Size of the axes title
    'axes.labelsize': 30,    # Size of the x and y labels
    'xtick.labelsize': 22,   # Size of the x tick labels
    'ytick.labelsize': 22,   # Size of the y tick labels
    'legend.fontsize': 30,   # Size of the legend font
    'figure.titlesize': 30   # Size of the figure title
})
def plot_attention(attentions, ticks, box, save_path):
    data = {}
    for i in range(len(ticks)):
            data[ticks[i]] = pd.Series(attentions[:,i], index=ticks)
    df = pd.DataFrame(data)
    fig, ax = plt.subplots(figsize=(25, 20))
    heatmap = sns.heatmap(df, cmap="viridis", annot=False, ax=ax)
    # Increase color bar text size
    cbar = heatmap.collections[0].colorbar
    cbar.ax.tick_params(labelsize=30)  # Font size of color bar ticks
    ax.tick_params(
        axis='both',          # changes apply to both x and y axes
        which='both',         # both major and minor ticks are affected
        bottom=False,         # ticks along the bottom edge are off
        top=False,            # ticks along the top edge are off
        left=False,           # ticks along the left edge are off
        right=False,          # ticks along the right edge are off
        labelbottom=False,    # labels along the bottom edge are off
        labelleft=False       # labels along the left edge are off
    )
    ax.set_xlabel("Key")
    ax.set_ylabel("Query")
    ax.set_title("Attention Heatmap")

    # Inset with zoom
    # axins = zoomed_inset_axes(ax, 1, loc='upper right')  # zoom=2
    axins = inset_axes(ax, width="130%", height="130%", loc='upper left', 
                   bbox_to_anchor=(0.3, 0.5, 0.5, 0.5), bbox_transform=ax.transAxes)

    sns.heatmap(df, cmap="viridis", annot=False, cbar=False, ax=axins,xticklabels=True, yticklabels=True)
    
    # Specify the coordinate bounds for the inset
    x1, x2, y1, y2 = box
    # x1, x2, y1, y2 = 120, 150, 120, 150 #20, 44, 62, 86  # Customize these values as needed
    axins.set_xlim(x1, x2)
    axins.set_ylim(y2, y1)  # Reversed the y-axis

    axins.tick_params(axis='both', which='both', colors='white')
    bbox = axins.get_position()
    rect = Rectangle((bbox.x0, bbox.y0), bbox.width, bbox.height, transform=fig.transFigure,
                     linewidth=2, edgecolor='white', facecolor='none')
    fig.patches.append(rect)
    rect_main = Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='white', facecolor='none', transform=ax.transData)
    ax.add_patch(rect_main)
    main_box_coords = [[x1, y1], [x2, y1], [x1, y2], [x2, y2]]
    
    inset_box_coords = [(bbox.x0, bbox.y0), (bbox.x0 + bbox.width, bbox.y0),
                        (bbox.x0, bbox.y0 + bbox.height), (bbox.x0 + bbox.width, bbox.y0 + bbox.height)]
    coords_pairs = [
        (main_box_coords[0], inset_box_coords[2]),  # Bottom-left of main to top-left of inset
        (main_box_coords[3], inset_box_coords[1]),  # Top-right of main to bottom-right of inset
    ]
    for main_coord, inset_coord in coords_pairs:
        main_coord_disp = ax.transData.transform(main_coord)
        inset_coord_disp = fig.transFigure.transform(inset_coord)
        
        (main_x, main_y) = fig.transFigure.inverted().transform(main_coord_disp)
        (inset_x, inset_y) = fig.transFigure.inverted().transform(inset_coord_disp)
        
        line = plt.Line2D([main_x, inset_x], [main_y, inset_y],
                          transform=fig.transFigure, color='white', linewidth=2)
        fig.lines.append(line)
    plt.savefig(save_path)
    plt.close()

def text_attention(input_ids, attentions, tokenizer, box):
        x1, x2, y1, y2 = box
        res = ""
        for i in range(y1, y2):
            idss = input_ids[0:i+1]
            attn = attentions[i][0:i+1]
            attn = (attn / attn.sum())*100
            latex_code = generate(idss, attn, tokenizer, color='red')
            # print("latex_code=", latex_code)
            res += f"({i}: {tokenizer.decode(input_ids[i])})\n"
            res += latex_code + "\n"
        return res
        # for i in range(y1, y2):
        #     for j in range(x1, x2):
        #         idss = input_ids[0:j+1]
        #         attn = attentions[i][0:j+1]
        #         attn = (attn / attn.sum())*100
        #         latex_code = generate(idss, attn, tokenizer, color='red')
        #         # print("latex_code=", latex_code)
        #         res += f"({i+1}: {tokenizer.decode(input_ids[i+1])}, {j+1}: {tokenizer.decode(input_ids[j+1])})\n"
        #         res += latex_code + "\n"
        # return res

class My_Dataset(Dataset):
    def __init__(self, dict):
        self.haha = dict
    def __len__(self):
        return len(self.haha["labels"])
    def __getitem__(self, idx):
        # print("self.haha['input_ids'][idx]=", self.haha["input_ids"])
        return {"input_ids": self.haha["input_ids"][idx], 
                "labels": self.haha["labels"][idx], 
                "attention_mask": self.haha["attention_mask"][idx],
                "full_trace": self.haha["full_trace"][idx],
                # "input_ids_right": self.haha["input_ids_right"][idx], 
                # "labels_right": self.haha["labels_right"][idx],
                # "attention_mask_right": self.haha["attention_mask_right"][idx],
                # "full_trace_right": self.haha["full_trace_right"][idx]
                }

class My_Dataset_2(Dataset):
    def __init__(self, dict):
        self.hihi = dict
    def __len__(self):
        return len(self.hihi["input"])
    def __getitem__(self, idx):
        res = {"input": self.hihi["input"][idx], "label": self.hihi["label"][idx], "ids": self.hihi["ids"][idx]}
        # print("res=", res)
        return res 

class mlp_Dataset(Dataset):
    def __init__(self, dict):
        self.hi = dict
    def __len__(self):
        return len(self.hi["label"])
    def __getitem__(self, idx):
        return {"input": self.hi["input"][idx], "label": self.hi["label"][idx]}

def get_mlp_dataset(window_size, tokenized_dataset, tokenizer):
    xs = None
    ys = None
    print("get_mlp_dataset")
    this_ds_inputs = torch.tensor(tokenized_dataset["input_ids"])
    this_ds_labels = copy.deepcopy(this_ds_inputs)
    this_mask = torch.tensor(tokenized_dataset["label"])
    this_ds_labels = this_ds_labels * this_mask + tokenizer.eos_token_id * (1-this_mask)
    # print("tokenized_dataset['input_ids']=", tokenized_dataset["input_ids"])
    for j in tqdm(range(1, this_ds_labels.size(1))):
        non_rows = torch.nonzero(this_ds_labels[:, j] != tokenizer.eos_token_id).transpose(0,1)
        non_rows = non_rows[0]
        # print("non_rows.size()=", non_rows.size())
        if non_rows.size(0) == 0:
             continue
        # print("non_rows=", non_rows)
        tmp_labels = this_ds_labels[non_rows]
        tmp_inputs = this_ds_inputs[non_rows]
        # tmp = this_ds[non_rows]
        tmp_xs = torch.ones(tmp_inputs.size(0), window_size) * tokenizer.eos_token_id
        cut_len = min(j, window_size)
        tmp_xs[:, -cut_len:] = tmp_inputs[:, j-cut_len:j]
        tmp_ys = copy.deepcopy(tmp_labels[:, j])
        if xs is None:
            xs = tmp_xs
            ys = tmp_ys
        else:
            xs = torch.cat((xs, tmp_xs), 0)
            ys = torch.cat((ys, tmp_ys), 0)
        # print("xs.size()=", xs.size())
        # print("ys.size()=", ys.size())
        # print("xs=", xs)
        # print("ys=", ys)
    # dict = {"input": xs, "label": ys}
    # print("dict=", dict)
    return mlp_Dataset({"input": xs, "label": ys})

def prepare_training_data(goal_graph, args, tokenizer, data_dir, type_dir):
    keep_dir = f"{type_dir}"
    print("keep_dir=", keep_dir)
    if not os.path.exists(keep_dir):
        os.makedirs(keep_dir)
    if not os.path.exists(f"{keep_dir}/train.pkl"):
        ICL_train_traces, ICL_train_label_masks,  ml1 = goal_graph.generate_structure_icl_data(num_traces=args.num_icl_train_traces,
                                max_examples=args.max_examples)
        # print("ICL_train_traces:", ICL_train_traces)
        
        ICL_valid_traces, ICL_valid_label_masks, ml2 = goal_graph.generate_structure_icl_data(num_traces=args.num_icl_valid_traces,
                                max_examples=args.max_examples)
        mk_train_traces, mk_train_label_masks, ml3 = goal_graph.generate_mk_data(num_traces=args.num_mk_train_traces,
                        max_examples=args.max_examples,
                        max_child_chain_len = args.max_child_chain_len)
        mk_valid_traces, mk_valid_label_masks,  ml4 = goal_graph.generate_mk_data(num_traces=args.num_mk_valid_traces,
                        max_examples=args.max_examples,
                        max_child_chain_len = args.max_child_chain_len)
        context_len = max(ml1, ml2, ml3, ml4)+10
        def tokenize_func(examples):
            input_ids = [ inn+ [tokenizer.pad_token_id] * (context_len - len(inn)) for inn in examples["ids"]]
            attention_mask = [[1] * len(inn) + [0] * (context_len - len(inn)) for inn in examples["ids"]]
            label_mask = [ inn + [0]* (context_len - len(inn)) for inn in examples["labmsk"]]
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "label": label_mask,
            }
        Train_traces = ICL_train_traces + mk_train_traces
        Train_label_masks = ICL_train_label_masks + mk_train_label_masks
        Train_mapping = {"ids": Train_traces, "labmsk": Train_label_masks}

        Train_ds = Dataset.from_dict(Train_mapping)
        Train_tokenized_datasets = Train_ds.map(tokenize_func, batched=True, remove_columns=["ids", "labmsk"])
        pkl.dump(Train_tokenized_datasets, open(f"{keep_dir}/train.pkl", "wb"))

        Valid_traces = ICL_valid_traces + mk_valid_traces
        Valid_label_masks = ICL_valid_label_masks + mk_valid_label_masks
        Valid_mapping = {"ids": Valid_traces, "labmsk": Valid_label_masks}
        Valid_ds = Dataset.from_dict(Valid_mapping)
        Valid_tokenized_datasets = Valid_ds.map(tokenize_func, batched=True, remove_columns=["ids", "labmsk"])
        pkl.dump(Valid_tokenized_datasets, open(f"{keep_dir}/valid.pkl", "wb"))
    return pkl.load(open(f"{keep_dir}/train.pkl", "rb")), pkl.load(open(f"{keep_dir}/valid.pkl", "rb"))

def do_test(goal_graph, model, tokenizer, test_max_examples, test_len, logger=None):
    res = {"whole": [], "final": [], "te_ver": [], "te_val":[], "node": [], "tr_val": [], "tr_ver": []}
    for i in range(test_max_examples): #MK_path_bounds[0][-1]-MK_path_bounds[0][0]+1
        if logger is not None:
            logger.info(f"shots num: {i}")
        accs = goal_graph.test_compositional_accs(test_len, model, i, 50, tokenizer)
        whole_acc = accs["whole"][0]
        fin_tok_acc = accs["final"][0]
        letter_acc = accs["letters"][0]
        if logger is not None:
            logger.info(f"whole acc={whole_acc}")
            logger.info(f"fin_tok_acc={fin_tok_acc}")
            logger.info(f"letter_acc={letter_acc}")
        te_val_accs = goal_graph.test_te_val_accs(test_len, model, i, 1000, tokenizer)
        te_val_acc = te_val_accs["val"][0]
        if logger is not None:
            logger.info(f"te_val_acc={te_val_acc}")
        mk_accs = goal_graph.test_mk_accs(2, model, i, 100, tokenizer)
        mk_node_acc = mk_accs["node"][0]
        mk_val_acc = mk_accs["val"][0]
        if logger is not None:
            logger.info(f"mk_node_acc={mk_node_acc}")
            logger.info(f"mk_val_acc={mk_val_acc}")
        icl_accs = goal_graph.test_icl_accs(2, model, i, 100, tokenizer)
        icl_acc = icl_accs["icl"][0]
        if logger is not None:
            logger.info(f"icl_acc={icl_acc}")
        res["whole"].append(whole_acc)
        res["final"].append(fin_tok_acc)
        res["te_ver"].append(letter_acc)
        res["te_val"].append(te_val_acc)
        res["node"].append(mk_node_acc)
        res["tr_val"].append(mk_val_acc)
        res["tr_ver"].append(icl_acc)

    tr_ver = np.mean(res["tr_ver"][1:])
    tr_val = np.mean(res["tr_val"])
    test_ver_0 = res["te_ver"][0]
    test_ver_f = max(res["te_ver"][1:])
    test_val_0 = res["te_val"][0]
    test_val_f = max(res["te_val"][1:])
    test_final_0 = res["final"][0]
    test_final_f = max(res["final"][1:])
    test_whole_0 = res["whole"][0]
    test_whole_f = max(res["whole"][1:])
    if not logger == None:
        logger.info(f"train vertices acc:{tr_ver}, train value acc:{tr_val}, \
    test vertices 0:{test_ver_0}, test vertices f:{test_ver_f}, \
    test value 0:{test_val_0}, test value f:{test_val_f}, \
    test_final_0:{test_final_0}, test_final_f:{test_final_f}, \
    test_whole_0:{test_whole_0}, test_whole_f:{test_whole_f}")
    return res

def do_probe(goal_graph, model, tokenizer, test_max_examples, max_child_len, test_len, probe_mean_num, logger, device, mode, typi):
    from utils.trainers import ProbeTrainer
    for shot_num in range(0,test_max_examples):
        logger.info(f"shot_num={shot_num}")
        parent_acc = []
        others_acc = []
        self_acc = []
        for pp in tqdm(range(probe_mean_num)):
            logger.info(f"iteration={pp}")
            if mode == "test":
                child_chain = random.choice(goal_graph.all_child_chains[test_len-1])
                chain_len = len(child_chain)
                dps = list(range(chain_len))
            elif mode == "mk":
                senmap = goal_graph.generate_mk_senmap(max_child_len)
                dps = senmap["my_dps"]
            for i, test_pos in enumerate(dps):
                    if test_pos == 0:
                        continue
                    for knock_pos in range(test_pos+1): 
                            eval_acc = []
                            if i == 0 or knock_pos != dps[i-1]:
                                    eps = random.random()
                                    if eps < 0.7:
                                        continue
                            if mode == "test" and knock_pos == dps[i-1]:
                                    eps = random.random()
                                    if eps < 0.7:
                                            continue
                            if mode == "test":
                                train_ds, test_ds = goal_graph.generate_test_probing_ds(child_chain, test_pos, knock_pos, shot_num, 100, model, typi)
                            elif mode == "mk":
                                train_ds, test_ds = goal_graph.generate_mk_probing_ds(senmap, test_pos, knock_pos, shot_num, 100, model, typi)
                            input_size = train_ds.__getitem__(0)["input"].size(0)
                            output_size = len(tokenizer)
                            linear_model = probing_MLP(input_size, output_size).to(device)
                            optimizer = torch.optim.Adam(linear_model.parameters(), lr=0.001)
                            trainer = ProbeTrainer(linear_model, optimizer)
                            dataloader = DataLoader(train_ds, batch_size=50, shuffle=True)
                            eval_dataloader = DataLoader(test_ds, batch_size=50, shuffle=True)
                            eval_acc += [trainer.train(dataloader, eval_dataloader,100, if_print=False)]
                            if i > 0 and knock_pos == dps[i-1]:
                                parent_acc += eval_acc
                            elif knock_pos == test_pos:
                                self_acc += eval_acc
                            else:
                                others_acc += eval_acc
            print("running_parent_acc_mean=", np.mean(parent_acc), "running_parent_acc_std=", np.std(parent_acc))
            print("running_others_acc_mean=", np.mean(others_acc), "running_others_acc_std=", np.std(others_acc))
        parent_acc_mean = np.mean(parent_acc)
        parent_acc_std = np.std(parent_acc)
        others_acc_mean = np.mean(others_acc)
        others_acc_std = np.std(others_acc)
        logger.info(f"the result of shot_num={shot_num}")
        logger.info(f"parent_acc_mean={parent_acc_mean}, parent_acc_std={parent_acc_std}")
        logger.info(f"others_acc_mean={others_acc_mean}, others_acc_std={others_acc_std}")

def do_plot(args, goal_graph, model, tokenizer, test_max_examples, test_len, device, train_ds, outs_path, test_epoch):
    child_chain = random.choice(goal_graph.all_child_chains[test_len-1])
    begin_pos = child_chain[0]["child_pos"]
    test_attn_pt = f"{outs_path}/test_attn.pkl"
    if not os.path.exists(test_attn_pt):
        input_ids_test = []
        for i in range(test_max_examples):
            beg_val = goal_graph.nodes[begin_pos[0]][begin_pos[1]].get_a_val()
            trace_str_prompt = goal_graph.draw_child_chain_trace(child_chain, beg_val)["trace_full"]
            input_ids_test += trace_str_prompt + tokenizer.encode("\n")
        pkl.dump(input_ids_test, open(test_attn_pt, "wb"))
    else:
         input_ids_test = pkl.load(open(test_attn_pt, "rb"))
    input_ids_icl = train_ds["input_ids"][2]
    # find the first eos
    for i in range(len(input_ids_icl)):
        if input_ids_icl[i] == tokenizer.eos_token_id:
            input_ids_icl = input_ids_icl[:i+1]
            break
    # input_ids_icl = input_ids_icl[0:100]
    input_ids_mk = train_ds["input_ids"][-7]
    for i in range(len(input_ids_mk)):
        if input_ids_mk[i] == tokenizer.eos_token_id:
            input_ids_mk = input_ids_mk[:i+1]
            break
    # input_ids_mk = input_ids_mk[0:100]
    print(tokenizer.decode(input_ids_mk))
    
    outputs_test = model(torch.tensor(input_ids_test).to(model.device), output_attentions=True)
    outputs_mk = model(torch.tensor(input_ids_mk).to(model.device), output_attentions=True)
    box =  (18, 48, 18, 48) #(18, 48, 62, 92)   #(28, 59, 28, 59)  #(28, 59, 119, 150)
    for l in range(args.n_layers):
        if l > 1:
            break
        for outputs, name in zip([outputs_test,  outputs_mk], ["test",  "mk"]):
            if name == "test":
                 box = (28, 59, 28, 59) if l == 0 else (28, 59, 119, 150)
            elif name == "icl":
                 box = (18, 48, 18, 48) if l == 0 else (18, 48, 62, 92)
            plot_dir = f"{outs_path}/plot_{name}_epoch{test_epoch}"
            if not os.path.exists(plot_dir):
                    os.makedirs(plot_dir)
            input_ids = input_ids_test if name=="test" else input_ids_icl if name=="icl" else input_ids_mk
            # print(outputs.attentions[l][:,1].size())
            attentions = torch.mean(outputs.attentions[l][0], dim=0).squeeze().detach().cpu().numpy()
            print(attentions.shape)
            # attentions = outputs.attentions[l][0][h].squeeze().detach().cpu().numpy()
            ticks = [f"[{pos}]:{tokenizer.decode(ids)}" for (pos, ids) in enumerate(input_ids)]
            plot_attention(attentions, ticks, box, f"{plot_dir}/check_layer{l}.png")
            latex_codes = text_attention(input_ids, attentions, tokenizer, box)
            with open(f"{plot_dir}/latex_layer{l}.txt", "w") as f:
                f.write(latex_codes)