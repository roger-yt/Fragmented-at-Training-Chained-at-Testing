import os
import logging
import torch
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
from transformers import AutoTokenizer, Trainer, TrainingArguments
from utils.collators import My_collator
from utils.configs import MyGPT2Config
from data_structure_related.data_structure import Goal_graph
from utils.networks import MyGPT2LMHeadModel
from utils.utils import prepare_training_data_hydra, do_test, do_probe, do_plot

log = logging.getLogger(__name__)

# def get_dirs(cfg):
#     data_dir = f"data_and_models/{cfg.experiment}/depth{cfg.graph.len}_maxchild{cfg.data.max_child_chain_len}/type{cfg.graph.type}"
#     if not os.path.exists(data_dir):
#             os.makedirs(data_dir)
#     model_dir = f"{data_dir}/outs_{cfg.model.name}"
#     if not os.path.exists(model_dir):
#         print("Create model_dir:", model_dir)
#         os.makedirs(model_dir)
#     outs_path = f"{model_dir}/layer{cfg.model.n_layers}_head{cfg.model.n_heads}_hidden{cfg.model.hidden_size}"
#     return data_dir, outs_path

@hydra.main(config_path="configs", config_name="config_normal")
def main(cfg: DictConfig):
    # print full config
    log.info(OmegaConf.to_yaml(cfg))

    # build Graph_shape
    gs = [cfg.graph.width] * cfg.graph.merge_pos + [1] + [cfg.graph.width] * (cfg.graph.len - cfg.graph.merge_pos - 1)
    assert cfg.graph.len > cfg.data.max_child_chain_len

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # data prep
    gg = Goal_graph(
        graph_shape=gs,
        graph_type=cfg.graph.type,
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
    # data_dir, outs_path = get_dirs(cfg)
    train_ds, valid_ds = prepare_training_data_hydra(gg, cfg, tokenizer)

    # print(cfg.trainer.output_dir)
if __name__ == "__main__":
    main()
