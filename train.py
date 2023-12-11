import torch
import argparse
import os
from torch.utils.data import Dataset, DataLoader
import sys
import torch
import torch.nn as nn
from torch.nn import functional as F
import pickle
from enum import Enum
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
import argparse
import json
from typing import Tuple, Optional, Union
import timeit
import ipdb
import logging
import yaml
import wandb
import time
import random
import evaluate
from utils.helper_functions import *#open_pickle, dump_pickle, save_model, Summary, AverageMeter, Metrics,int2mil
from data.cocodataset import *
from utils.eval_utils import validation, language_eval
from utils.train_algos import LMCriterion, SCST
from utils.rewards import init_scorer

TRAIN = False
torch.manual_seed(0)
random.seed(0)
torch.autograd.set_detect_anomaly(True)

def train(model, config):

    # params and model
    model_name = config["wandb"]["run_name"]
    val_min = float(1000)
    metric_max = 0
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    batch_size = config['batch_size']
    train_bsz = config['batch_size']
    epochs = config['num_epochs']
    output_dir = config['out_dir'] 

    if config['train_method']=='mle':
        train_bsz = config['mle_bsz']
        optimizer = AdamW(model.parameters(), lr=float(config['lr']))

    else:
        load_model(model, output_dir,f'clip_mle_best_cider')
        optimizer = AdamW(model.parameters(), lr=float(2e-5))
        init_scorer(config['cached_tokens'])

    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    #load pretrained model if scst

    model = model.to(device)
    model.train()
    if config['freeze_gpt']:
        model.gpt.eval()
    
    loss_meter = AverageMeter("train_loss", ":.5f")

    # Dataloaders
    
    if TRAIN:
        train_dataset = CocoDataset(config['train_data'], config['prefix_length'],config['normalize_prefix'], 'train', config['tokenizer'])
        train_dataloader = DataLoader(train_dataset, batch_size=train_bsz, shuffle=True, drop_last=True, num_workers = 4)
        
        if config['train_method']=="mle":
            scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=config['warmup_steps'], num_training_steps=epochs* len(train_dataloader))
        else:
            scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=0, num_training_steps=epochs* len(train_dataloader))
  


    val_dataset = CocoDataset(config['val_data'], config['prefix_length'],config['normalize_prefix'],'val', config['tokenizer'])
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_dataset = CocoDataset(config['test_data'], config['prefix_length'],config['normalize_prefix'],'test', config['tokenizer'])
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    tokenizer = val_dataset.tokenizer
    prefix_len = val_dataset.prefix_len
    max_length = val_dataset.max_len_token
    temp = config['temp']
    stop_token =  tokenizer.encode(config['stop_token'])[0]

    step = 1

    # Validation loss before epoch 1
    if config['init_val']:
        val_loss_meter, val_lang_stats = validation(model, val_dataloader,val_dataset, device, config)
    
    #### "val_loss_avg": val_loss_meter.avg,
        val_log = {"val_CIDEr" : val_lang_stats["CIDEr"],
                "val_SPICE" : val_lang_stats["SPICE"]
                                }
        if config['logging']: 
            wandb.log(val_log, step = step)
        print(val_log)

    for epoch in range(epochs):
        
        print(f">>> Training epoch {epoch}")
        sys.stdout.flush()

        # progress = tqdm(total=len(train_dataloader), desc=output_prefix)
        epoch_train_losses = []
        epoch_train_decoded_cap = []
        epoch_train_tgts = []
        train_start = time.time()

        predictions = [] # coco
        step_time_avg = []

        for idx, (prefix, targets, mask, untokenized_cap, meta_data) in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            # if idx == 100:
            #     break
            # step_time_start = time.time()

            model.zero_grad()
            optimizer.zero_grad()            
            targets = targets.view(-1, targets.shape[-1])
            mask = mask.view(-1, mask.shape[-1])
            
            targets, mask, prefix = targets.to(device), mask.to(device), prefix.to(device, dtype=torch.float32) # (B,41), (B,51), (B,1024/512)

            if config['train_method'] == 'mle':
                prefix = repeat_tensors(targets.shape[0]//prefix.shape[0],prefix)
                loss, preds, entropy, perplexity = LMCriterion(model, prefix, targets, mask, meta_data, prefix_len)
                
                # Decode batch preds and add it to coco_predictions list
                if config['log_train_metrics'] :  
                    decoded_cap = tokenizer.batch_decode(preds)
                    
                    for k, sent in enumerate(decoded_cap):
                        entry = {'image_id' : meta_data['cocoid'][k].item(), 'caption': sent, 'perplexity': perplexity[k].item(), 'entropy': entropy[k].item()}
                        predictions.append(entry)
                    
                    #scst optimizes on metrics. Tracking their train performance makes no sense.
                    epoch_train_tgts.append(targets)
                    epoch_train_decoded_cap.extend(decoded_cap)

            else:
                reward, loss = SCST(model, prefix, targets, mask, max_length, stop_token,tokenizer, config)
                
            # accumulating loss
            epoch_train_losses.append(loss.item())
            loss_meter.update(loss.item(), targets.shape[0])
            loss.backward()
            optimizer.step()
            scheduler.step()
            step+=1

            #logging step info
            train_log = {"epoch": epoch+1,
            "train_loss_avg": loss_meter.avg,
            "lr": optimizer.state_dict()["param_groups"][0]["lr"],
            
            }
            
            print(train_log)
            # step_time_avg.append(time.time() - step_time_start)
            # print(f"batch {config['batch_size']} sample_n {config['train_sample_n']}  time avg : {np.mean(np.array(step_time_avg))}")

            if config['logging']:
                wandb.log(train_log, step = step)
        

        #Eval metrics for epoch i
        if config['log_train_metrics'] and config['train_method'] == 'mle':
            epoch_train_tgts = torch.cat((epoch_train_tgts), dim=0)
            mask = epoch_train_tgts > 0
            target_cap  = [[tokenizer.decode(epoch_train_tgts[i][mask[i]])] for i in range(epoch_train_tgts.shape[0])]
            train_lang_stats = language_eval("cocotalk.json", predictions, "train_temp", 'train')
    
            train_log = {"train_CIDEr" : train_lang_stats["CIDEr"],"train_SPICE" : train_lang_stats["SPICE"]}

        #Validation
        val_start = time.time()
        val_loss_meter, val_lang_stats = validation(model, val_dataloader, val_dataset, device, config)
   
        val_log = {"val_CIDEr" : val_lang_stats["CIDEr"],
                "val_SPICE" : val_lang_stats["SPICE"]
                }
        val_end = time.time()
        print(f'train time : {val_start - train_start} val time : {val_end - val_start}')
        
        # Logging epoch info 
        if config['logging']: 
            wandb.log(val_log, step = step)
            # logging.info({**train_log, **val_log})
        # print({**train_log, **val_log})

        #Saving
        # if config['logging'] and config['save_best_val']:
        #         if val_loss_meter.avg< val_min:
        #             val_min = val_loss_meter.avg
        #             save_model(output_dir,f'{model_name}_epoch_{epoch+1}',model)
        if config['logging'] and config['save_best_metric']:
                if val_lang_stats["CIDEr"] > metric_max:
                    metric_max  = val_lang_stats["CIDEr"]
                    save_model(output_dir,f'{model_name}_best_cider',model)
                    
        elif config['logging'] and config['save_every_epoch']:
            save_model(output_dir,f'{epoch+1}',model)
            
    return model

def trigger_training(config):
        
    model = Model(clip_dim = config['prefix_dim'], prefix_len = config['prefix_length'], const_len =config['prefix_length'], 
                num_layers = config['num_layers'], attn_heads = config['attn_heads'], freeze_gpt = config['freeze_gpt'],cocotalk = config['cocotalk'])
    trainable_params(model)

    if config['logging'] and (not config["wandb"]["sweep"]):
        wandb.init(entity=config["wandb"]["entity"], project=config["wandb"]["project"], config=config)
        wandb.run.name = config["wandb"]["run_name"]
    
    train(model,config)

def sweep_agent_manager():
    wandb.init()
    config = dict(wandb.config)
    run_name = config["wandb"]["run_name"]
    wandb.run.name = run_name
    # logging.basicConfig(filename=f'/home2/manugaur/clip_cap_manu/logs/num_layer_sweep/{run_name}.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    trigger_training(config)

def get_config():
    with open('./configs/clip_cap.yml') as f:
        config = yaml.load(f,Loader=yaml.FullLoader)
    
    set_data_dir(config)
    return config

def main():
    config = get_config()
    if config['wandb']['sweep']:
        wandb.agent(sweep_id=f"manugaur/clip_cap_reproduce/{config['wandb']['sweep_id']}", function=sweep_agent_manager, count=config['wandb']['sweep_run_count'])
    else:
        trigger_training(config)

if __name__ == '__main__':
    main()

"""

 Why self.caption_tokens[idx] = tokens ::::: here padding is -1

"""
