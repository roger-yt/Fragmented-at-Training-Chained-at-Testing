import random
from datasets import Dataset, DatasetDict, concatenate_datasets
import re
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
import torch
import copy
import os
import pickle as pkl
from utils.utils import My_Dataset, My_Dataset_2
from demonstration.probing_related import *
import re

Alphabet = [i for i in range(32, 58)] + [i for i in range(64, 90)]

class Node():
    def __init__(self, pos=None, ids=None, val=None, potential_ranges=None, children_edges=None, parents_edges=None):
        self.pos = pos
        self.ids = ids
        self.children_edges = []
        self.parents_edges = []
        self.potential_ranges = []
    
    def add_child_edge(self, op, op_val, child_pos):
        self.children_edges.append({"op": op, "op_val": op_val, "child_pos": child_pos})
    
    def add_parent_edge(self, op, op_val, parent_pos): 
        self.parents_edges.append({"op": op, "op_val": op_val, "parent_pos": parent_pos})
    
    def show_all_features(self):
        print("pos:", self.pos)
        print("ids:", self.ids)
        print("potential_ranges:", self.potential_ranges)
        print("children_edges:", self.children_edges)
        print("parents_edges:", self.parents_edges)
    
    def get_all_potential_vals(self):
        tmp_list = []
        for ran in self.potential_ranges:
            tmp_list += list(range(ran[0], ran[1]))
        tmp_list = list(set(tmp_list))
        return tmp_list
    def get_a_val(self):
        tmp_list = self.get_all_potential_vals()
        return random.choice(tmp_list)

def deal_with(op, op_val, val):
    if op == "+":
        return int(val + op_val)
    elif op == "-":
        return int(val - op_val)
    elif op == "*":
        return int(val * op_val)
    else:
        return int(val / op_val)
    
def produce_chain_nodes(idss, opas, beg_range):
    assert len(idss) == len(opas) + 1, "The length of letters must be one more than the length of opas"
    nodes = []
    for i in range(len(idss)):
        this_node = Node(pos=(i, 0), ids=idss[i])
        if i == 0:
            this_node.potential_ranges.append(beg_range)
        nodes.append([this_node])
    for i in range(len(idss)-1):
        if i < len(idss)-1:
            nodes[i][0].add_child_edge(opas[i][0], opas[i][1], (i+1, 0))
            nodes[i+1][0].add_parent_edge(opas[i][0], opas[i][1], (i, 0))
            nodes[i+1][0].potential_ranges += [(deal_with(opas[i][0], opas[i][1], range[0]), deal_with(opas[i][0], opas[i][1], range[1])) for range in nodes[i][0].potential_ranges]
    return nodes

def correct_potential_ranges(nodes):
    for i in range(1, len(nodes)):
        for j in range(len(nodes[i])):
            for k in range(len(nodes[i][j].parents_edges)):
                edge = nodes[i][j].parents_edges[k]
                parent_pos = edge["parent_pos"]
                op = edge["op"]
                op_val = edge["op_val"]
                nodes[i][j].potential_ranges += [(deal_with(op, op_val, range[0]), deal_with(op, op_val, range[1])) for range in nodes[parent_pos[0]][parent_pos[1]].potential_ranges]
            nodes[i][j].potential_ranges = list(set(nodes[i][j].potential_ranges))
    return nodes

def cross_two_tree_nodes(nodes1, nodes2, depth):
    assert len(nodes1) == len(nodes2), "The depth of two nodes must be the same"
    new_nodes = copy.deepcopy(nodes1)
    for i in range(len(new_nodes)):
        if not i == depth:
            this_pos = (i, len(new_nodes[i]))
            this_ids = copy.deepcopy(nodes2[i][0].ids)
            this_ranges = copy.deepcopy(nodes2[i][0].potential_ranges) if i==0 else []
            # print("this_letter:", this_letter)
            # print("this_ranges:", this_ranges)
            this_node = Node(pos=this_pos, ids=this_ids)
            this_node.potential_ranges = this_ranges
            # this_node.show_all_features()
            new_nodes[i].append(this_node)
    # for i in range(len(new_nodes)):
    #     for j in range(len(new_nodes[i])):
    #         new_nodes[i][j].show_all_features()
    for i in range(len(new_nodes)):
        if i < len(new_nodes)-1:
            this_child_op = copy.deepcopy(nodes2[i][0].children_edges[0]["op"])
            this_child_op_val = copy.deepcopy(nodes2[i][0].children_edges[0]["op_val"])
            new_nodes[i][-1].add_child_edge(this_child_op, this_child_op_val, (i+1, len(new_nodes[i+1])-1))
        if i > 0:
            this_parent_op = copy.deepcopy(nodes2[i][0].parents_edges[0]["op"])
            this_parent_op_val = copy.deepcopy(nodes2[i][0].parents_edges[0]["op_val"])
            new_nodes[i][-1].add_parent_edge(this_parent_op, this_parent_op_val, (i-1, len(new_nodes[i-1])-1))
    return correct_potential_ranges(new_nodes)

def save_nodes(nodes, rangi, typi):
    shape = []
    for i in range(len(nodes)):
        print(i)
        shape.append(len(nodes[i]))
    shape_str = "-".join([str(x) for x in shape])
    save_dir = f"nodes_dir/shape{shape_str}_range_{rangi[0]}-{rangi[1]}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    type_dir = f"{save_dir}/type{typi}"
    if not os.path.exists(type_dir):
        os.makedirs(type_dir)
    path = f"{type_dir}/nodes.pkl"
    if not os.path.exists(path):
        pkl.dump(nodes, open(path, "wb"))

def generate_random_setof_nodes(alphabet, depth, width, rangi, typi):
    used_letters = []
    dns = []
    for i in range(width):
        alpha_chain = random.sample(list(set(alphabet)-set(used_letters)), depth)
        op_chain = [(random.choice(["+", "-"]), random.choice(range(1, 10))) for _ in range(depth-1)]
        dn = produce_chain_nodes(alpha_chain, op_chain, rangi)
        if width == 1:
            save_nodes(dn, rangi, typi)
        dns.append(dn)
        used_letters += alpha_chain
    
    if width > 1:
        for i in range(0, int(depth/2)):
            crossed = cross_two_tree_nodes(dns[0], dns[1], i)
            for j in range(2, width):
                crossed = cross_two_tree_nodes(crossed, dns[j], i)
            save_nodes(crossed, rangi, typi)

def child_chains_from_pos(nodes, pos, length):
    # deep first search
    assert pos[0] + length <= len(nodes), "The child chain is too long"
    start_edge = {"op": None, "op_val": None, "child_pos": pos}
    res = [] # list of chains
    if length == 0:
        return [[start_edge]]
        # for edge in nodes[pos[0]][pos[1]].children_edges:
        #     res.append([start_edge, edge])
        # return res
    for edge in nodes[pos[0]][pos[1]].children_edges:
        child_pos = edge["child_pos"]
        child_chains = child_chains_from_pos(nodes, child_pos, length-1)
        for chain in child_chains:
            chain[0]["op"] = edge["op"]
            chain[0]["op_val"] = edge["op_val"]
            res.append([start_edge]+chain)
    return res

def get_all_chains(nodes):
    res = [[] for _ in range(len(nodes))] # res[i] is the chain whose edge length is i
    for i in range(len(nodes)):
        for j in range(len(nodes[i])):
            for le in range(len(nodes)-i):
                chains = child_chains_from_pos(nodes, (i, j), le)
                res[le] += chains
    return res

class Goal_graph():
    def __init__(self, graph_shape, graph_type, vocab_size, env_val_num_low, chain_val_num, leak_prob_node, leak_prob_val, 
                 addlen, nearlen, tl_low, 
                 context_lower, context_upper, context_div, tokenizer):
        self.vocab_size = vocab_size
        self.env_val_num_low = env_val_num_low # L-l
        self.chain_val_num = chain_val_num # l
        self.alphabet = copy.deepcopy(Alphabet)[:vocab_size] # N
        self.operations = ["+", "-", "*", "/"]
        self.tokenizer = tokenizer
        self.begin_range = (100, 100 + chain_val_num) # l
        self.graph_shape = graph_shape # n
        self.graph_type = graph_type
        self.leak_prob_node = leak_prob_node # mu
        self.leak_prob_val = leak_prob_val # eps
        self.graph_depth = len(graph_shape)
        self.graph_width = max([graph_shape[i] for i in range(self.graph_depth)])
        self.context_lower = context_lower
        self.context_upper = context_upper
        self.context_div = context_div
        self.nodes, self.all_env_vals, self.all_token_idss,\
            self.letter_context, self.all_child_chains = self.__get_nodes__()
        # self.all_env_vals = list(range(0, 0+self.env_val_num_low))
        self.alal_token_ids = self.__get_alal_token_ids__()
        self.all_nodes_ids_to_pos = self.__get_all_nodes_ids_to_pos__()
        # self.all_env_vals = self.__get_all_env_vals__() # L
        self.bigg_mo = 1000
        self.addlen = addlen
        self.nearlen = nearlen
        self.tl_low = tl_low
    
    def __get_token_idss__(self, nodes, all_env_vals):
        biaodian = ['\n', ',',  '<|endoftext|>', '=', '?:', ]
        token_letters = [self.tokenizer.encode(ele)[0] for ele in biaodian] + self.alphabet
        num_token_letters = [self.tokenizer.encode(str(val))[0] for val in all_env_vals]
        for i in range(len(nodes)):
            for j in range(len(nodes[i])):
                vals = nodes[i][j].get_all_potential_vals()
                for val in vals:
                    num_token_letters.append(self.tokenizer.encode(str(val))[0])
        num_token_letters = list(set(num_token_letters))
        num_token_letters.sort()
        token_letters += num_token_letters
        return token_letters
    
    def __get_letter_context__(self, all_token_idss):
        # print("self.all_token_letters:", self.all_token_letters)
        used_tokens = [ids for ids in all_token_idss]
        # print("used_tokens:", used_tokens)
        unused_tokens = list(set(range(len(self.tokenizer))) - set(used_tokens))
        res = {}
        for letter in self.alphabet:
            res[letter] = random.sample(unused_tokens, self.context_div)
            unused_tokens = list(set(unused_tokens) - set(res[letter]))
        return res

    def __get_all_env_vals__(self, nodes, noise_range):
        exclude = []
        for i in range(len(nodes)):
            for j in range(len(nodes[i])):
                exclude += nodes[i][j].get_all_potential_vals()
        res = list(range(noise_range[0], noise_range[1]))
        res = list(set(res) - set(exclude))
        # res += list(range(noise_range[0], noise_range[1]))
        res = list(set(res))
        return res
    
    def __get_nodes__(self):
        shape_str = "-".join([str(i) for i in self.graph_shape])
        nodes_dir = f"nodes_dir/shape{shape_str}_range_{self.begin_range[0]}-{self.begin_range[1]}/type{self.graph_type}"
        nodes_path = f"{nodes_dir}/nodes.pkl"
        print(nodes_path)
        if not os.path.exists(nodes_path):
            print("generate nodes")
            generate_random_setof_nodes(self.alphabet, self.graph_depth, self.graph_width, self.begin_range, self.graph_type)
        if os.path.exists(nodes_path):
            print("find nodes in the directory")
            nodes = pkl.load(open(nodes_path, "rb"))
            noise_range = (0, 0+self.env_val_num_low)
            nodes_related_path = f"{nodes_dir}/related_cl{self.context_lower}_cu{self.context_upper}_\
cd{self.context_div}_nl{noise_range[0]}_nu{noise_range[1]}.pkl"
            if os.path.exists(nodes_related_path):
                rel_dict = pkl.load(open(nodes_related_path, "rb"))
            else:
                rel_dict = {}
                all_env_vals = self.__get_all_env_vals__(nodes, noise_range)
                all_token_idss = self.__get_token_idss__(nodes, all_env_vals)
                letter_context = self.__get_letter_context__(all_token_idss)
                rel_dict["all_env_vals"] = all_env_vals
                rel_dict["all_token_idss"] = all_token_idss
                rel_dict["letter_context"] = letter_context
                pkl.dump(rel_dict, open(nodes_related_path, "wb"))
            all_child_chain_path = f"{nodes_dir}/all_child_chains.pkl"
            if os.path.exists(all_child_chain_path):
                all_child_chains = pkl.load(open(all_child_chain_path, "rb"))
            else:
                all_child_chains = get_all_chains(nodes)
                pkl.dump(all_child_chains, open(all_child_chain_path, "wb"))
            return nodes, rel_dict["all_env_vals"], rel_dict["all_token_idss"], rel_dict["letter_context"], all_child_chains
        
    def __get_alal_token_ids__(self):
        a_ids = [ids for ids in self.all_token_idss]
        for key in self.letter_context:
            a_ids += self.letter_context[key]
        a_ids = list(set(a_ids))
        return a_ids
        
    def __get_all_nodes_ids_to_pos__(self):
        res = {}
        for i in range(len(self.nodes)):
            for j in range(len(self.nodes[i])):
                res[self.nodes[i][j].ids] = (i, j)
        return res
    
    def draw_trace(self, idss, vals):
        part_trace_wo_value = []
        part_trace_w_value = []
        chain_wo_value = []
        chain_w_value = []
        trace_ques = [idss[-1]] + self.tokenizer.encode("=?:")
        trace_len = len(idss)
        trace_chain = []
        for i in range(0, trace_len):
            trace_chain += random.sample(self.letter_context[idss[i]], random.choice(range(self.context_lower, self.context_upper)))
            trace_chain += [idss[i]] + self.tokenizer.encode("=")
            part_trace_wo_value.append(trace_ques+trace_chain)
            chain_wo_value.append(trace_chain)
            if i != trace_len - 1:
                trace_chain += self.tokenizer.encode(str(vals[i])) +self.tokenizer.encode(",")
                chain_w_value.append(trace_chain)
                part_trace_w_value.append(trace_ques+trace_chain)
            else:
                trace_chain += self.tokenizer.encode(str(vals[i]))
                part_trace_w_value.append(trace_ques+trace_chain)
                chain_w_value.append(trace_chain)
        
        labels = part_trace_w_value[-1][len(part_trace_w_value[0]):]
        trace_map = {"ls_wo_value": part_trace_wo_value, "ls_w_value": part_trace_w_value, "trace_full": part_trace_w_value[-1], "trace_head":part_trace_w_value[0],
                    "trace_ques": trace_ques, "trace_chain": trace_chain,
                "ls_chain_wo_value": chain_wo_value, "ls_chain_w_value": chain_w_value, "nodes_idss": idss, 
                "vals": vals, "labels": labels}
        label_mask = [0]*len(trace_map["trace_head"]) + [1]*(len(trace_map["trace_full"]) - len(trace_map["trace_head"]))
        trace_map["label_mask"] = label_mask
        return trace_map
    
    def generate_vals(self, ops, op_vals, begin_val, modu): # no redundant op
        vals = [begin_val]
        for i in range(0, len(ops)):
            this_op = ops[i]
            this_val = op_vals[i]
            if this_op == "+":
                vals.append(int(abs(vals[-1] + this_val)) % modu)
            elif this_op == "-":
                vals.append(int(abs(vals[-1] - this_val)) % modu)
            elif this_op == "*":
                vals.append(int(abs(vals[-1] * this_val)) % modu)
            else:
                vals.append(int(abs(vals[-1] / this_val)) % modu)
        return vals
    
    def generate_trace(self, idss, ops, op_vals, begin_val, modu=1000):
        assert len(idss) == len(ops) + 1, "The length of letters must be one more than the length of ops"
        vals = self.generate_vals(ops, op_vals, begin_val, modu)
        trace_map = self.draw_trace(idss, vals)
        return trace_map
    
    def generate_one_normal_example(self, max_examples):
        trace_length = random.randint(2, self.graph_depth+self.addlen)
        all_corpus = copy.deepcopy(self.alphabet)
        all_related = list(self.all_nodes_ids_to_pos.keys())
        sen = random.sample(all_corpus, trace_length) #random.sample(list(set(all_corpus)-set(all_related)), trace_length)
        trace_str = []
        label_mask = []
        all_inter_vals = []
        for _ in range(max_examples):
            node_vals = [random.choice(self.all_env_vals) for _ in range(trace_length)]
            all_inter_vals.append(node_vals)
            this_trace_map = self.draw_trace(sen, node_vals)
            trace_str += this_trace_map["trace_full"]
            label_mask += this_trace_map["label_mask"]
            trace_str += self.tokenizer.encode("\n")
            label_mask += [0]
        return {"trace_str": trace_str, "label_mask": label_mask, "sen": sen, "all_inter_vals": all_inter_vals}
    
    def generate_structure_icl_data(self, num_traces, max_examples):
        traces = []
        label_masks = []
        max_len = 0
        for _ in tqdm(range(num_traces)):
            example_map = self.generate_one_normal_example(max_examples)
            trace_str = example_map["trace_str"]
            label_mask = example_map["label_mask"]
            traces.append(trace_str)
            label_masks.append(label_mask)
            max_len = max(max_len, len(trace_str))
        return traces, label_masks, max_len
    
    def generate_mk_senmap(self, max_child_chain_len):
        tl_lower_bound = max(max_child_chain_len, self.tl_low)
        tl_upper_bound = max(tl_lower_bound+1, self.graph_depth+self.addlen)
        trace_length = random.randint(tl_lower_bound, tl_upper_bound)
        expected_child_chain_edge_len = random.randint(1, max_child_chain_len-1)
        child_chain = copy.deepcopy(random.choice(self.all_child_chains[expected_child_chain_edge_len]))
        node_idss = [self.nodes[edge["child_pos"][0]][edge["child_pos"][1]].ids for edge in child_chain]
        node_ops = [edge["op"] for edge in child_chain[1:]]
        node_op_vals = [edge["op_val"] for edge in child_chain[1:]]
        
        diam = min(len(child_chain)+self.nearlen, trace_length)
        my_dps = sorted(random.sample(range(diam), len(child_chain)))
        added = random.choice(range(trace_length-diam+1))
        my_dps = [x+added for x in my_dps]
        # print("my_dps:", my_dps)
        all_corpus = copy.deepcopy(self.alphabet)
        all_related = list(self.all_nodes_ids_to_pos.keys())
        sen = []
        for i in range(trace_length):
            if i in my_dps:
                sen.append(node_idss[my_dps.index(i)])
            else:
                eps = random.uniform(0, 1)
                if eps < self.leak_prob_node:
                    sen.append(random.choice(list(set(all_related)-set(node_idss)-set(sen))))
                else:
                    sen.append(random.choice(list(set(all_corpus)-set(all_related)-set(node_idss)-set(sen))))
        return {"sen":sen, "child_chain": child_chain, "my_dps": my_dps, "node_idss": node_idss, "node_ops": node_ops, "node_op_vals": node_op_vals}
    
    def generate_mk_tracemap(self, senmap):
        child_chain = senmap["child_chain"]
        node_ops = senmap["node_ops"]
        node_op_vals = senmap["node_op_vals"]
        my_dps = senmap["my_dps"]
        sen = senmap["sen"]
        trace_length = len(sen)
        all_related = list(self.all_nodes_ids_to_pos.keys())
        beg_pos = child_chain[0]["child_pos"]
        begin_val = self.nodes[beg_pos[0]][beg_pos[1]].get_a_val()
        chain_vals = self.generate_vals(node_ops, node_op_vals, begin_val, self.bigg_mo)
        inter_vals = []
        for i in range(trace_length):
            if i in my_dps:
                #find the pos of i in my_dps
                i_idx = my_dps.index(i)
                inter_vals.append(chain_vals[i_idx])
            else:
                if sen[i] in all_related:
                    pos = self.all_nodes_ids_to_pos[sen[i]]
                    in_vals = self.nodes[pos[0]][pos[1]].get_all_potential_vals()
                    out_vals = list(set(self.all_env_vals) - set(in_vals))
                    eps = random.uniform(0, 1)
                    if eps < self.leak_prob_val:
                        inter_vals.append(random.choice(in_vals))
                    else:
                        inter_vals.append(random.choice(out_vals))
                else:
                    inter_vals.append(random.choice(self.all_env_vals))
        this_trace_map = self.draw_trace(sen, inter_vals)
        return this_trace_map

    def generate_one_mk_example(self, max_child_chain_len, max_examples):
        senmap = self.generate_mk_senmap(max_child_chain_len)
        trace_str = []
        all_inter_vals = []
        label_mask = []
        for _ in range(max_examples):
            this_trace_map = self.generate_mk_tracemap(senmap)
            all_inter_vals.append(this_trace_map["vals"])
            trace_str += this_trace_map["trace_full"]
            label_mask += this_trace_map["label_mask"]
            trace_str += self.tokenizer.encode("\n")
            label_mask += [0]
        return {"trace_str": trace_str, "label_mask": label_mask, "child_chain": senmap["child_chain"], "sen": senmap["sen"], "all_inter_vals": all_inter_vals, "dps": senmap["my_dps"]}
    
    def generate_mk_data(self, num_traces, max_examples, max_child_chain_len):
        assert max_child_chain_len >=2, "max_child_chain_len must be greater than or equal to 2"
        if self.graph_depth > 2:
            assert max_child_chain_len < self.graph_depth, "max_child_chain_len must be less than graph depth"
        traces = []
        label_masks = []
        max_len = 0
        for _ in tqdm(range(num_traces)):
            example_map = self.generate_one_mk_example(max_child_chain_len, max_examples)
            # print("trace_str:", self.tokenizer.decode(example_map["trace_str"]))
            # print("dps:", example_map["dps"])
            trace_str = example_map["trace_str"]
            label_mask = example_map["label_mask"]
            max_len = max(max_len, len(trace_str))
            traces.append(trace_str)
            label_masks.append(label_mask)
        return traces, label_masks, max_len
    
    def draw_child_chain_trace(self, child_chain, beg_val):
        node_ops = [edge["op"] for edge in child_chain[1:]]
        node_op_vals = [edge["op_val"] for edge in child_chain[1:]]
        node_idss = [self.nodes[edge["child_pos"][0]][edge["child_pos"][1]].ids for edge in child_chain]
        trace_map = self.generate_trace(node_idss, node_ops, node_op_vals, beg_val, self.bigg_mo)
        return trace_map

    def generate_compositional_tests(self, test_path_len, shot_num, chain_num, rep_per_chain):
        xs = []
        labels = []
        ys = []
        full_trs = []
        context_len = 0
        for _ in range(chain_num):
            child_chain = random.choice(self.all_child_chains[test_path_len-1])
            root_node_pos = child_chain[0]["child_pos"]
            alll_beg_vals = self.nodes[root_node_pos[0]][root_node_pos[1]].get_all_potential_vals()
            for _ in range(rep_per_chain):
                trace_examples = []
                for _ in range(shot_num):
                    trace_str_prompt = self.draw_child_chain_trace(child_chain, random.choice(alll_beg_vals))["trace_full"]
                    trace_examples += trace_str_prompt + self.tokenizer.encode("\n")
                que_map = self.draw_child_chain_trace(child_chain, random.choice(alll_beg_vals))
                que_head = que_map["trace_head"]
                que_full = que_map["trace_full"]
                que_label = que_map["labels"]
                xs.append(trace_examples + que_head)
                ys.append(que_label)
                full_trs.append(trace_examples + que_full)
                context_len = max(context_len, len(xs[-1])+len(ys[-1]))
        input_ids = [[self.tokenizer.pad_token_id] * (context_len - len(inn)) + inn for inn in xs]
        labels = [[self.tokenizer.pad_token_id] * (context_len - len(inn)) + inn for inn in ys]
        attention_mask = [[0] * (context_len - len(inn)) + [1]*len(inn) for inn in xs]
        full_trs = [[self.tokenizer.pad_token_id] * (context_len - len(inn)) + inn for inn in full_trs]

        return My_Dataset({"input_ids": torch.tensor(input_ids), "labels": torch.tensor(labels),
                            "full_trace": torch.tensor(full_trs), "attention_mask": torch.tensor(attention_mask)})

    def test_compositional_accs(self, test_path_len, model, shot_num, total_num, tokenizer, batch_size=50, sample_times=10):
        sum_correct_whole = 0
        sum_correct_final =0
        sum_correct_letters = 0
        whole_acc_examples = []
        final_acc_examples = []
        letters_acc_examples = []
        ds = self.generate_compositional_tests(test_path_len, shot_num, sample_times, total_num)
        dataloader = DataLoader(ds, batch_size=batch_size, shuffle=True)
        num_data = len(dataloader.dataset)
        for batch in tqdm(dataloader):
            prompts = batch["input_ids"].to(next(model.parameters()).device)
            targets = batch["labels"].to(next(model.parameters()).device)
            attention_masks = batch["attention_mask"].to(next(model.parameters()).device)
            new_tokens_len = (test_path_len-1)*(self.context_upper + 5)
            outputs = model.generate(prompts, max_new_tokens=new_tokens_len, pad_token_id=tokenizer.eos_token_id, attention_mask=attention_masks)
            outputs = outputs[:, prompts.size(1):]
            _predicts = tokenizer.batch_decode(outputs, skip_special_tokens=True)

            for i in range(len(outputs)):
                pattern = r'(\w=\d+)'
                targets_matches = re.findall(pattern, tokenizer.decode(targets[i]))
                predicts_matches = re.findall(pattern, _predicts[i])
                
                tar_pt = 0
                tar_pt_let = 0
                correct_final = False
                for j in range(len(predicts_matches)):
                    if tar_pt < len(targets_matches) and predicts_matches[j] == targets_matches[tar_pt]:
                        tar_pt += 1
                    if tar_pt_let < len(targets_matches) and predicts_matches[j][0] == targets_matches[tar_pt_let][0]:
                        tar_pt_let += 1
                for j in range(len(predicts_matches)-1, -1, -1):
                    if predicts_matches[j][0] == targets_matches[-1][0]:
                        correct_final = (predicts_matches[j] == targets_matches[-1])
                        break

                if tar_pt == len(targets_matches):
                    sum_correct_whole += 1
                    correct_final = True
                    if len(whole_acc_examples) < 50 and shot_num > 0:
                        whole_acc_examples.append(self.tokenizer.decode(batch["full_trace"][i]))
                if correct_final:
                    sum_correct_final += 1
                    if len(final_acc_examples) < 50 and shot_num > 0:
                        final_acc_examples.append(self.tokenizer.decode(batch["full_trace"][i]))
                if tar_pt_let == len(targets_matches):
                    sum_correct_letters += 1
                    if len(letters_acc_examples) < 50 and shot_num > 0:
                        letters_acc_examples.append(batch["full_trace"][i])
                
        return {"whole": (sum_correct_whole/num_data, whole_acc_examples),
                "final": (sum_correct_final/num_data, final_acc_examples),
                "letters": (sum_correct_letters/num_data, letters_acc_examples)}

    def generate_mk_tests(self, max_child_chain_len, shot_num, total_num):
        node_xs = []
        node_ys = []
        val_xs = []
        val_ys = []
        full_trace = []
        context_len = 0
        for _ in range(total_num):
            mk_map = self.generate_one_mk_example(max_child_chain_len, shot_num+1)
            mk_child_chain = mk_map["child_chain"]
            mk_sen = mk_map["sen"]
            mk_dps = mk_map["dps"]
            mk_inter_vals = mk_map["all_inter_vals"]
            trace_examples = []
            for i in range(shot_num):
                trace_examples += self.draw_trace(mk_sen, mk_inter_vals[i])["trace_full"] + self.tokenizer.encode("\n")
            que_map = self.draw_trace(mk_sen, mk_inter_vals[shot_num])
            node_trace_examples = trace_examples + que_map["ls_w_value"][mk_dps[0]]
            node_xs.append(node_trace_examples)
            next_num_tok = self.tokenizer.encode(str(mk_inter_vals[shot_num][mk_dps[1]]))[0]
            node_ys.append([mk_sen[mk_dps[1]], next_num_tok])

            full_trace.append(trace_examples + que_map["trace_full"])
            context_len = max(context_len, len(full_trace[-1]))

            val_trace_examples = trace_examples + que_map["ls_wo_value"][mk_dps[1]]
            val_xs.append(val_trace_examples)
            val_ys.append(next_num_tok)
            
        node_input_ids = [[self.tokenizer.pad_token_id] * (context_len - len(inn)) + inn for inn in node_xs]
        node_attention_mask = [[0] * (context_len - len(inn)) + [1]*len(inn) for inn in node_xs]
        val_input_ids = [[self.tokenizer.pad_token_id] * (context_len - len(inn)) + inn for inn in val_xs]
        val_attention_mask = [[0] * (context_len - len(inn)) + [1]*len(inn) for inn in val_xs]
        full_trace = [[self.tokenizer.pad_token_id] * (context_len - len(inn)) + inn for inn in full_trace]
        
        node_ds = My_Dataset({"input_ids": torch.tensor(node_input_ids), "labels": torch.tensor(node_ys),
                            "full_trace": torch.tensor(full_trace), "attention_mask": torch.tensor(node_attention_mask)})
        val_ds = My_Dataset({"input_ids": torch.tensor(val_input_ids), "labels": torch.tensor(val_ys),
                            "full_trace": torch.tensor(full_trace), "attention_mask": torch.tensor(val_attention_mask)})
        
        return node_ds, val_ds

    def test_mk_accs(self, max_child_chain_len, model, shot_num, total_num, tokenizer, batch_size=50):
        node_ds, val_ds = self.generate_mk_tests(max_child_chain_len, shot_num, total_num)
        node_dataloader = DataLoader(node_ds, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(val_ds, batch_size=batch_size, shuffle=True)
        node_num_data = len(node_dataloader.dataset)
        node_sum_correct = 0
        val_num_data = len(val_dataloader.dataset)
        val_sum_correct = 0
        print("test node_ds")
        for batch in tqdm(node_dataloader):
            prompts = batch["input_ids"].to(next(model.parameters()).device)
            labels = batch["labels"].to(next(model.parameters()).device)
            attention_masks = batch["attention_mask"].to(next(model.parameters()).device)
            new_tokens_len = (self.graph_depth+self.addlen-1)*(self.context_upper + 5)
            # print("prompts.size():", prompts.size())
            outputs = model.generate(prompts, max_new_tokens=new_tokens_len, pad_token_id=tokenizer.eos_token_id, attention_mask=attention_masks)
            outputs = outputs[:, prompts.size(1):]
            _predicts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            # print("prompts:", prompts[0:1])
            # print("prompts:", prompts[0:1])
            # print("prompts:", self.tokenizer.batch_decode(prompts[0:1], skip_special_tokens=True))
            # print("labels:", self.tokenizer.batch_decode(labels[0:1], skip_special_tokens=True))
            # print("predicts:", _predicts[0:1])
            # print("targets:", targets[0:1])
            for i in range(len(outputs)):
                delim_pos = _predicts[i].find("\n")
                predict = _predicts[i][:delim_pos]
                letter = self.tokenizer.decode(labels[i][0])
                # print("predict:", predict)
                # print("letter:", letter)
                pattern = r'(\w=\d+)'
                matches = re.findall(pattern, predict)
                # print("matches:", matches)
                for match in matches:
                    if match[0] == letter:
                        node_sum_correct += 1
                        break
        
        print("test val_ds")
        for batch in tqdm(val_dataloader):
            prompts = batch["input_ids"].to(next(model.parameters()).device)
            labels = batch["labels"].to(next(model.parameters()).device)
            attention_masks = batch["attention_mask"].to(next(model.parameters()).device)
            new_tokens_len = (self.graph_depth+self.addlen-1)*(self.context_upper + 5)
            outputs = model.generate(prompts, max_new_tokens=1, pad_token_id=tokenizer.eos_token_id, attention_mask=attention_masks)
            outputs = outputs[:, prompts.size(1):][:,0]
            val_sum_correct += torch.sum(outputs == labels).item()
            # _predicts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            # print("prompts:", self.tokenizer.batch_decode(prompts[0:1], skip_special_tokens=True))
            # print("labels:", self.tokenizer.batch_decode(labels[0:1], skip_special_tokens=True))
            # print("predicts:", _predicts[0:1])
        
        return {"node": (node_sum_correct/node_num_data, None),
                "val": (val_sum_correct/val_num_data, None)}

    def generate_te_val_tests(self, test_path_len, shot_num, total_num):
        val_xs = []
        val_ys = []
        full_trace = []
        context_len = 0
        for _ in range(total_num):
            child_chain = random.choice(self.all_child_chains[test_path_len-1])
            root_node_pos = child_chain[0]["child_pos"]
            alll_beg_vals = self.nodes[root_node_pos[0]][root_node_pos[1]].get_all_potential_vals()
            trace_examples = []
            for _ in range(shot_num):
                trace_str_prompt = self.draw_child_chain_trace(child_chain, random.choice(alll_beg_vals))["trace_full"]
                trace_examples += trace_str_prompt + self.tokenizer.encode("\n")
            que_map = self.draw_child_chain_trace(child_chain, random.choice(alll_beg_vals))
            test_pos = random.choice(range(1, len(child_chain)))
            val_trace_examples = trace_examples + que_map["ls_wo_value"][test_pos]
            lab = self.tokenizer.encode(str(que_map["vals"][test_pos]))[0]
            # print("lab:", lab)
            # print("que_map[ls_wo_value][test_pos]:", self.tokenizer.decode(que_map["ls_wo_value"][test_pos]))
            # print("que_map[ls_w_value][test_pos]:", self.tokenizer.decode(que_map["ls_w_value"][test_pos]))
            

            full_trace.append(trace_examples + que_map["trace_full"])
            context_len = max(context_len, len(full_trace[-1]))

            val_xs.append(val_trace_examples)
            val_ys.append(lab)
        # print("context_len:", context_len)
        # for inn in node_xs:
        #     print("len(inn):", len(inn))
        val_input_ids = [[self.tokenizer.pad_token_id] * (context_len - len(inn)) + inn for inn in val_xs]
        val_attention_mask = [[0] * (context_len - len(inn)) + [1]*len(inn) for inn in val_xs]
        full_trace = [[self.tokenizer.pad_token_id] * (context_len - len(inn)) + inn for inn in full_trace]
        
        val_ds = My_Dataset({"input_ids": torch.tensor(val_input_ids), "labels": torch.tensor(val_ys),
                            "full_trace": torch.tensor(full_trace), "attention_mask": torch.tensor(val_attention_mask)})
        
        return val_ds
    def test_te_val_accs(self, test_path_len, model, shot_num, total_num, tokenizer, batch_size=50):
        val_ds = self.generate_te_val_tests(test_path_len, shot_num, total_num)
        val_dataloader = DataLoader(val_ds, batch_size=batch_size, shuffle=True)
        val_num_data = len(val_dataloader.dataset)
        val_sum_correct = 0
        for batch in tqdm(val_dataloader):
            prompts = batch["input_ids"].to(next(model.parameters()).device)
            labels = batch["labels"].to(next(model.parameters()).device)
            attention_masks = batch["attention_mask"].to(next(model.parameters()).device)
            new_tokens_len = (self.graph_depth+self.addlen-1)*(self.context_upper + 5)
            # print("prompts:", self.tokenizer.batch_decode(prompts[0:1], skip_special_tokens=True))
            outputs = model.generate(prompts, max_new_tokens=1, pad_token_id=tokenizer.eos_token_id, attention_mask=attention_masks)
            outputs = outputs[:, prompts.size(1):][:,0]
            # print("outputs:", outputs)
            # print("labels:", labels)
            val_sum_correct += torch.sum(outputs == labels).item()
            # _predicts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            # print("prompts:", self.tokenizer.batch_decode(prompts[0:1], skip_special_tokens=True))
            # print("labels:", self.tokenizer.batch_decode(labels[0:1], skip_special_tokens=True))
            # print("predicts:", _predicts[0:1])
        
        return {"val": (val_sum_correct/val_num_data, None)}


    def generate_icl_tests(self, max_child_chain_len, shot_num, total_num):
        xs = []
        ys = []
        full_trace = []
        context_len = 0
        y_len = 0
        for _ in range(total_num):
            eps = random.uniform(0, 1)
            if eps < 0.5:
                t_map = self.generate_one_normal_example(shot_num+1)
            else:
                t_map = self.generate_one_mk_example(max_child_chain_len, shot_num+1)
            sen = t_map["sen"]
            inter_vals = t_map["all_inter_vals"]
            trace_examples = []
            for i in range(shot_num):
                trace_examples += self.draw_trace(sen, inter_vals[i])["trace_full"] + self.tokenizer.encode("\n")
            que_map = self.draw_trace(sen, inter_vals[shot_num])
            xs.append(trace_examples + que_map["trace_head"])
            ys.append(sen[1:])
            y_len = max(y_len, len(sen[1:]))
            full_trace.append(trace_examples + que_map["trace_full"])
            context_len = max(context_len, len(full_trace[-1]))
        input_ids = [[self.tokenizer.pad_token_id] * (context_len - len(inn)) + inn for inn in xs]
        labels = [[self.tokenizer.pad_token_id] * (y_len - len(inn)) + inn for inn in ys]
        attention_mask = [[0] * (context_len - len(inn)) + [1]*len(inn) for inn in xs]
        full_trs = [[self.tokenizer.pad_token_id] * (context_len - len(inn)) + inn for inn in full_trace]

        return My_Dataset({"input_ids": torch.tensor(input_ids), "labels": torch.tensor(labels),
                            "full_trace": torch.tensor(full_trs), "attention_mask": torch.tensor(attention_mask)})
    
    def test_icl_accs(self, max_child_chain_len, model, shot_num, total_num, tokenizer, batch_size=50):
        ds = self.generate_icl_tests(max_child_chain_len, shot_num, total_num)
        dataloader = DataLoader(ds, batch_size=batch_size, shuffle=True)
        num_data = len(dataloader.dataset)
        sum_correct = 0
        acc_examples = []
        for batch in tqdm(dataloader):
            prompts = batch["input_ids"].to(next(model.parameters()).device)
            targets = batch["labels"].to(next(model.parameters()).device)
            attention_masks = batch["attention_mask"].to(next(model.parameters()).device)
            new_tokens_len = (self.graph_depth+self.addlen-1)*(self.context_upper + 5)
            outputs = model.generate(prompts, max_new_tokens=new_tokens_len, pad_token_id=tokenizer.eos_token_id, attention_mask=attention_masks)
            outputs = outputs[:, prompts.size(1):]
            _predicts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            # print("prompts:", self.tokenizer.batch_decode(prompts[0:1], skip_special_tokens=True))
            # print("labels:", self.tokenizer.batch_decode(targets[0:1], skip_special_tokens=True))
            # print("predicts:", _predicts[0:1])
            for i in range(len(outputs)):
                delim_pos = _predicts[i].find("\n")
                predict = _predicts[i][:delim_pos]
                target = self.tokenizer.decode(targets[i], skip_special_tokens=True)
                # print("target:", target)
                pattern = r'(\w=\d+)'
                matches = re.findall(pattern, predict)
                # print("matches:", matches)
                # print("target:", target)
                match_str = ""
                for i in range(min(len(matches), len(target))):
                    match_str += matches[i][0]
                # print("match_str:", match_str)
                if match_str == target:
                    sum_correct += 1
                # for i in range(len(target)):

                # if predict == target:
                #     sum_correct += 1
                #     if len(acc_examples) < 50 and shot_num > 0:
                #         acc_examples.append(self.tokenizer.decode(batch["full_trace"][i]))
        return {"icl": (sum_correct/num_data, acc_examples)}

    def test_blurred_accs(self, test_path_len, model, shot_num, total_num, tokenizer, batch_size=50, sample_times=10):
            num_data = 0
            sum_correct = 0
            acc_examples = []
            for _ in tqdm(range(sample_times)):
                beg_depth = random.randint(0, self.graph_depth-test_path_len)
                beg_pos = (beg_depth, random.choice(range(self.graph_shape[beg_depth])))
                child_chain = self.find_a_child_chain(beg_pos, test_path_len)
                beg_rel_dep = random.choice(range(len(child_chain)-1))
                ds = self.generate_blurred_tests(child_chain, beg_rel_dep, shot_num, total_num, tokenizer)
                dataloader = DataLoader(ds, batch_size=batch_size, shuffle=True)
                # get the number of data in dataloader
                num_data += len(dataloader.dataset)
                for batch in dataloader:
                    prompts = batch["ids"].to(next(model.parameters()).device)
                    targets = batch["label"].to(next(model.parameters()).device)
                    # print("prompts:", prompts[0:2])
                    # print("targets:", targets[0:2])
                    # print(tokenizer.batch_decode(prompts[0:3]))
                    # print(tokenizer.batch_decode(targets[0:3]))
                    # print("len(prompts):", [len(prompts[i]) for i in range(len(prompts))])
                    # catch the Value Error
                    # try:
                    #     prompts_ids = tokenizer(prompts, return_tensors="pt").input_ids.to(model.device)
                    # except ValueError:
                    #     ids_list = [tokenizer(prompts[i], return_tensors="pt").input_ids for i in range(len(prompts))]
                    #     print("sizes:", [ids_list[i].size() for i in range(len(ids_list))])
                    #     print("decode:", [[tokenizer.decode(ids_list[i][0][j])for j in range(len(ids_list[i][0]))] for i in range(len(ids_list))])

                    # print("prompts_ids.size():", prompts_ids.size())
                    new_tokens_len = 1
                    # print("new_tokens_len:", new_tokens_len)

                    outputs = model.generate(prompts, max_new_tokens=new_tokens_len, pad_token_id=tokenizer.eos_token_id)
                    # print("outputs.size():", outputs.size())
                    # print(tokenizer.batch_decode(outputs))
                    # print("end generate")
                    # print("outputs=", outputs)
                    # outputs = outputs[:, -new_tokens_len-7:]

                    # print("prompts:", prompts[0:1])
                    # print("predicts:", _predicts[0:1])
                    # print("targets:", targets[0:1])
                    for i in range(len(outputs)):
                        if outputs[i][-1] == targets[i]:
                            sum_correct += 1

                        
            return {"blurred": (sum_correct/num_data, acc_examples),}
    
    def generate_test_probing_ds(self, child_chain, test_pos, knock_pos, shot_num, total_num, model, typi, por_train=0.8):
        assert test_pos < len(child_chain), "The test position must be less than the length of the child chain"
        assert test_pos >= 1, "The test position must be greater than 1"
        # print("child_chain:", child_chain)
        root_node_pos = child_chain[0]["child_pos"]
        all_potential_beg_vals = self.nodes[root_node_pos[0]][root_node_pos[1]].get_all_potential_vals()
        beg_vals = []
        for _ in range(total_num):
            beg_vals.append(random.sample(all_potential_beg_vals, min(shot_num+1, len(all_potential_beg_vals))))
        xs = []
        ys = []
        idss = []
        context_len = 0
        for _ in range(total_num):
            trace_examples = []
            for _ in range(shot_num):
                trace_str_prompt = self.draw_child_chain_trace(child_chain, random.choice(all_potential_beg_vals))["trace_full"]
                trace_examples += trace_str_prompt + self.tokenizer.encode("\n")
            que_map = self.draw_child_chain_trace(child_chain, random.choice(all_potential_beg_vals))
            que_idss = copy.deepcopy(que_map["nodes_idss"])
            que_vals = copy.deepcopy(que_map["vals"])
            if typi == "val":
                knock_ids = que_idss[knock_pos]
                knock_node_pos = self.all_nodes_ids_to_pos[knock_ids]
                knock_node = self.nodes[knock_node_pos[0]][knock_node_pos[1]]
                repi = random.sample(knock_node.get_all_potential_vals(), 1)[0]
                que_vals[knock_pos] = repi
                lab = self.tokenizer.encode(str(repi))[0]
            elif typi == "node":
                all_related = list(self.all_nodes_ids_to_pos.keys())
                repi = random.choice(all_related)
                # repi = random.choice(copy.deepcopy(self.alphabet))
                que_idss[knock_pos] = repi
                lab = repi
            
            new_map = self.draw_trace(que_idss, que_vals)
            idss.append(trace_examples + new_map["ls_wo_value"][test_pos])
            context_len = max(context_len, len(idss[-1]))
            ys.append(lab)
            lasti = model(torch.tensor(idss[-1]).to(model.device)).last_hidden_state[-1, :].squeeze().detach().tolist()
            xs.append(lasti)
        train_num = int(por_train*len(xs))
        idss = [[self.tokenizer.pad_token_id] * (context_len - len(inn)) + inn for inn in idss]

        train_mapping = {"input": torch.tensor(xs[0:train_num]), "label": torch.tensor(ys[0:train_num]), "ids": torch.tensor(idss[0:train_num])}
        test_mapping = {"input": torch.tensor(xs[train_num:]), "label": torch.tensor(ys[train_num:]), "ids": torch.tensor(idss[train_num:])}
        return My_Dataset_2(train_mapping), My_Dataset_2(test_mapping)
    
    def generate_mk_probing_ds(self, senmap, test_pos, knock_pos, shot_num, total_num, model, typi, por_train=0.8):
        assert test_pos < len(senmap["sen"]), "The test position must be less than the length of the child chain"
        assert test_pos >= 1, "The test position must be greater than 1"
        # print("child_chain:", child_chain)
        xs = []
        ys = []
        idss = []
        context_len = 0
        # print("begin")
        for _ in range(total_num):
            trace_examples = []
            for _ in range(shot_num):
                trace_str_prompt = self.generate_mk_tracemap(senmap)["trace_full"]
                trace_examples += trace_str_prompt + self.tokenizer.encode("\n")
            que_map = self.generate_mk_tracemap(senmap)
            # print("que_map:", self.tokenizer.decode(que_map["trace_full"]))  
            que_idss = copy.deepcopy(que_map["nodes_idss"])
            que_vals = copy.deepcopy(que_map["vals"])
            knock_ids = que_idss[knock_pos]
            if knock_ids in self.all_nodes_ids_to_pos.keys():
                knock_node_pos = self.all_nodes_ids_to_pos[knock_ids]
                knock_node = self.nodes[knock_node_pos[0]][knock_node_pos[1]]
                rep = random.sample(knock_node.get_all_potential_vals(), 1)[0]
                que_vals[knock_pos] = rep
            else:
                rep = random.choice(self.all_env_vals)
                que_vals[knock_pos] = rep
            new_map = self.draw_trace(que_idss, que_vals)
            idss.append(trace_examples + new_map["ls_wo_value"][test_pos])
            context_len = max(context_len, len(idss[-1]))
            ys.append(self.tokenizer.encode(str(rep))[0])
            lasti = model(torch.tensor(idss[-1]).to(model.device)).last_hidden_state[-1, :].squeeze().detach().tolist()
            xs.append(lasti)
        train_num = int(por_train*len(xs))
        idss = [[self.tokenizer.pad_token_id] * (context_len - len(inn)) + inn for inn in idss]
        train_mapping = {"input": torch.tensor(xs[0:train_num]), "label": torch.tensor(ys[0:train_num]), "ids": torch.tensor(idss[0:train_num])}
        test_mapping = {"input": torch.tensor(xs[train_num:]), "label": torch.tensor(ys[train_num:]), "ids": torch.tensor(idss[train_num:])}
        return My_Dataset_2(train_mapping), My_Dataset_2(test_mapping)