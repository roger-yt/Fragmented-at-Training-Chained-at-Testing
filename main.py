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
from utils.utils import prepare_training_data, do_test, do_probe, do_plot_hydra
import pickle as pkl
import sys

log = logging.getLogger(__name__)

def get_dirs(cfg):
    # print("cfg.paths.data_dir=", cfg.paths.data_dir)
    model_dir = f"{cfg.paths.data_dir}/outs_{cfg.model.name}"
    if not os.path.exists(model_dir):
        print("Create model_dir:", model_dir)
        os.makedirs(model_dir)
    outs_path = f"{model_dir}/layer{cfg.model.n_layers}_head{cfg.model.n_heads}_hidden{cfg.model.hidden_size}"
    return outs_path

def get_latest_checkpoint(outs_path):
        checkpoints = [d for d in os.listdir(outs_path) if d.startswith("checkpoint-")]
        if len(checkpoints) == 0:
            return None, None
        latest_checkpoint = max(checkpoints, key=lambda x: int(x.split("-")[1]))
        return os.path.join(outs_path, latest_checkpoint), latest_checkpoint

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
    outs_path = get_dirs(cfg)
    train_ds, valid_ds = pkl.load(open(f"{cfg.paths.data_dir}/train.pkl", "rb")), pkl.load(open(f"{cfg.paths.data_dir}/valid.pkl", "rb"))

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

    print("outs_path:", outs_path)
    train_args = TrainingArguments(
        output_dir=outs_path,
        eval_strategy="epoch",
        num_train_epochs=cfg.trainer.num_train_epochs,
        save_steps=cfg.trainer.save_steps,
            per_device_eval_batch_size=cfg.trainer.per_device_eval_batch_size,                                                                                                               
            per_device_train_batch_size=cfg.trainer.per_device_train_batch_size,  
            report_to="none",
            gradient_accumulation_steps=cfg.trainer.gradient_accumulation_steps,
            learning_rate=cfg.trainer.learning_rate,
    )
    if cfg.modes.train:
        trainer = Trainer(
            model=model,
            tokenizer=tokenizer,
            args=train_args,
            data_collator=data_collator,
            train_dataset=train_ds,
            eval_dataset=valid_ds
            )

        latest_checkpoint, ckpt = get_latest_checkpoint(outs_path)
        if latest_checkpoint is not None:
                trainer.train(resume_from_checkpoint=True)
        else:
                trainer.train()
    if cfg.modes.test or cfg.modes.probe or cfg.modes.plot:
        if cfg.test.epoch==-1:
               checkpoint_dirs = [d for d in os.listdir(outs_path) if d.startswith("checkpoint-")]
               if checkpoint_dirs:
                        latest_checkpoint = max(checkpoint_dirs, key=lambda x: int(x.split("-")[1]))
                        model_path = os.path.join(outs_path, latest_checkpoint)
                        test_epoch = int(latest_checkpoint.split("-")[1])
               else:
                        raise FileNotFoundError(f"No checkpoint directories found in {outs_path}")
        else:
                model_path = f"{outs_path}/checkpoint-{cfg.test.epoch}"
                test_epoch = cfg.test.epoch
        model = MyGPT2LMHeadModel.from_pretrained(model_path, config=model_cfg).to(device)
        training_args = torch.load(f"{model_path}/training_args.bin")
        test_len = len(gs)
    
    if cfg.modes.test:
        trainer = Trainer(
            model=model,
            tokenizer=tokenizer,
            args=train_args,
            data_collator=data_collator,
            train_dataset=train_ds,
            eval_dataset=valid_ds
        )
        log_path = f"{outs_path}/test_epoch{test_epoch}_len{test_len}.log"
        print("log_path:", log_path)
        handler = logging.FileHandler(log_path, mode='w')
        handler.setFormatter(logging.Formatter('%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s'))
        logger = logging.getLogger('testing')
        logger.setLevel(logging.DEBUG)
        logger.addHandler(handler)
        logger.addHandler(logging.StreamHandler(sys.stdout))
        logger.info(trainer.evaluate())
        do_test(gg, model, tokenizer, cfg.data.max_examples, test_len, logger)
    
    if cfg.modes.probe:

        log_path = f"{outs_path}/prob_epoch{test_epoch}_meannum{cfg.probe.mean_num}.log"
        handler = logging.FileHandler(log_path, mode='w')
        handler.setFormatter(logging.Formatter('%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s'))
        logger = logging.getLogger('probing')
        logger.setLevel(logging.DEBUG)
        logger.addHandler(handler)
        logger.addHandler(logging.StreamHandler(sys.stdout))
        logger.info(f"path len={test_len}")

        logger.info("mk probing")
        do_probe(gg, model, tokenizer, cfg.data.max_examples, cfg.data.max_child_chain_len, test_len, cfg.probe.mean_num, logger, device, "mk", "val")
        
        logger.info("test probing")
        do_probe(gg, model, tokenizer, cfg.data.max_examples, cfg.data.max_child_chain_len, test_len, cfg.probe.mean_num, logger, device, "test", "val")
    if cfg.modes.plot:
        # print(test_epoch)
        do_plot_hydra(cfg, gg, model, tokenizer, 2, test_len,  device, train_ds, outs_path, cfg.test.epoch)
if __name__ == "__main__":
    main()
