import os
import logging
import torch
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
from transformers import AutoTokenizer, Trainer
from utils.collators import My_collator
from utils.configs import MyGPT2Config
from data_structure_related.data_structure import Goal_graph
from utils.networks import MyGPT2LMHeadModel
from utils.utils import prepare_training_data, do_test, do_probe, do_plot

log = logging.getLogger(__name__)

def get_dirs(cfg):
    data_dir = f"data_and_models"
    shape_dir = f"len{cfg.graph.len}_width{cfg.graph.width}_merge{cfg.graph.merge_pos}"
    if not os.path.exists(f"{data_dir}/{shape_dir}"):
            os.makedirs(f"{data_dir}/{shape_dir}")
    foot_str = f"{shape_dir}/maxchildlen{str(cfg.graph.max_child_chain_len)}\
_cl{cfg.data.context_lower}_cu{cfg.data.context_upper}_cd{cfg.data.context_div}_vocab{str(cfg.model.vocab_size)}_envaln{str(cfg.data.env_val_num_low)}\
_chainvaln{str(cfg.data.chain_val_num)}_lkpn{cfg.data.leak_prob_node}_lkpv{cfg.data.leak_prob_val}\
_addlen{cfg.data.addlen}_nearlen{cfg.data.nearlen}_tl{cfg.data.tl_low}_shot{cfg.data.max_examples}_icl{str(cfg.data.num_icl_train)}_mk{str(cfg.data.num_mk_train)}"


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):
    # print full config
    log.info(OmegaConf.to_yaml(cfg))

    # build Graph_shape
    gs = [cfg.graph.width] * cfg.graph.merge_pos + [1] + [cfg.graph.width] * (cfg.graph.len - cfg.graph.merge_pos - 1)
    assert cfg.graph.len > cfg.graph.max_child_chain_len

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
    train_ds, valid_ds = prepare_training_data(gg, cfg, tokenizer, "data_and_models", ".")

    # model & config
    context_len = 2048
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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MyGPT2LMHeadModel(model_cfg).to(device)
    data_collator = My_collator(tokenizer)

    # Trainer
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=cfg.trainer,
        data_collator=data_collator,
        train_dataset=train_ds,
        eval_dataset=valid_ds
    )

    # train / resume
    trainer.train()

    # testing, probing, plotting can follow similarly...
    # e.g. do_test(…); do_probe(…); do_plot(…)

if __name__ == "__main__":
    main()
