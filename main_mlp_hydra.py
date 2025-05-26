import os
import logging
import torch
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
from transformers import AutoTokenizer, Trainer, TrainingArguments
from utils.collators import My_collator
from utils.configs import MyMLPConfig
from data_structure_related.data_structure import Goal_graph
from utils.networks import MyMLP
from utils.utils import get_mlp_dataset, do_test, do_probe, do_plot_hydra
import pickle as pkl
import sys
from utils.trainers import MLPTrainer

log = logging.getLogger(__name__)

def get_dirs(cfg):
    # print("cfg.paths.data_dir=", cfg.paths.data_dir)
    model_dir = f"{cfg.paths.data_dir}/outs_{cfg.model.name}"
    if not os.path.exists(model_dir):
        print("Create model_dir:", model_dir)
        os.makedirs(model_dir)
    outs_path = f"{model_dir}/layer{cfg.model.n_layers}_hidden{cfg.model.hidden_size}"
    return outs_path

def get_latest_checkpoint(outs_path):
        checkpoints = [d for d in os.listdir(outs_path) if d.startswith("checkpoint-")]
        if len(checkpoints) == 0:
            return None, None
        latest_checkpoint = max(checkpoints, key=lambda x: int(x.split("-")[1]))
        return os.path.join(outs_path, latest_checkpoint), latest_checkpoint

@hydra.main(config_path="configs", config_name="config_mlp")
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
    outs_path = get_dirs(cfg)
    train_mlp_path = f"{cfg.paths.data_dir}/train_window{cfg.model.window_size}.pkl"
    valid_mlp_path = f"{cfg.paths.data_dir}/valid_window{cfg.model.window_size}.pkl"
    if not os.path.exists(train_mlp_path):
        train_ds, valid_ds = pkl.load(open(f"{cfg.paths.data_dir}/train.pkl", "rb")), pkl.load(open(f"{cfg.paths.data_dir}/valid.pkl", "rb"))
        train_ds = get_mlp_dataset(cfg.model.window_size, train_ds, tokenizer)
        valid_ds = get_mlp_dataset(cfg.model.window_size, valid_ds, tokenizer)
        pkl.dump(train_ds, open(train_mlp_path, "wb"))
        pkl.dump(valid_ds, open(valid_mlp_path, "wb"))
    train_ds, valid_ds = pkl.load(open(train_mlp_path, "rb")), pkl.load(open(valid_mlp_path, "rb"))
    for i in range(-100, -95):
        print(len(train_ds[i]["input"]))
        # print(Train_ds[i]["input"])
        print(tokenizer.decode(train_ds[i]["input"].int()))
        print()

    # model & config
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_cfg = MyMLPConfig(window_size = cfg.model.window_size,
        in_dim=cfg.model.window_size * len(gg.alal_token_ids), 
              out_dim=len(tokenizer), 
              hidden_dim=cfg.model.hidden_size, 
              layer_num=cfg.model.n_layers, 
              all_token_ids=gg.alal_token_ids, 
              vocab_size=len(tokenizer))
    model = MyMLP(model_cfg).to(device)

    from torch.utils.data.dataloader import DataLoader
    train_dataloader = DataLoader(train_ds, batch_size=10000, shuffle=True)
    eval_dataloader = DataLoader(valid_ds, batch_size=10000, shuffle=True)

    print("outs_path:", outs_path)
    if cfg.modes.train:
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-5, weight_decay=0.0001)
        trainer = MLPTrainer(model, optimizer)
        if (outs_path is not None) and (not os.path.exists(outs_path)):
            os.makedirs(outs_path)
        trainer.train(train_dataloader, eval_dataloader, cfg.trainer.num_train_epochs, outs_path)

    if cfg.modes.test:
        if cfg.test.epoch==-1:
               checkpoint_dirs = [d for d in os.listdir(outs_path) if d.startswith("checkpoint-")]
               if checkpoint_dirs:
                        latest_checkpoint = max(checkpoint_dirs, key=lambda x: int(x.split("-")[1].split(".")[0]))
                        max_epoch = int(latest_checkpoint.split("-")[1].split(".")[0])
                        model_path = os.path.join(outs_path, latest_checkpoint)
                        log_path = f"{outs_path}/test_epoch{max_epoch}.log"
               else:
                        raise FileNotFoundError(f"No checkpoint directories found in {outs_path}")
        else:
                model_path = f"{outs_path}/checkpoint-{cfg.test.epoch}.pt"
                log_path = f"{outs_path}/test_epoch{cfg.test.epoch}.log"
        print("log_path=", log_path)
        handler = logging.FileHandler(log_path, mode='w')
        handler.setFormatter(logging.Formatter('%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s'))
        logger = logging.getLogger('testing')
        logger.setLevel(logging.DEBUG)
        logger.addHandler(handler)
        logger.addHandler(logging.StreamHandler(sys.stdout))
        
        model.load_state_dict(torch.load(model_path)["model_state_dict"])
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
        trainer = MLPTrainer(model, optimizer)
        eval_acc, eval_loss = trainer.evaluate(eval_dataloader)
        logger.info(f"Eval Loss: {eval_loss}")
        test_len = len(gs)
        do_test(gg, model, tokenizer, cfg.data.max_examples, test_len, logger)
    
if __name__ == "__main__":
    main()
