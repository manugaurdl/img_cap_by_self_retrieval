import torch
import argparse
import os
from torch.utils.data import Dataset, DataLoader
import sys
import torch
import torch.nn as nn
from torch.nn import functional as nnf
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
import wandb
import yaml

use_log = True
save_ckpt = True


def open_pickle(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

def dump_pickle(data,path):
    with open(path, 'wb') as f:
        pickle.dump(data,f)

def init_wandb(args):

    config = {
    "lr": args.lr,
    "model": "RN50", # pretrained/scratch
    "num_epochs": args.epochs, # CHANGE
    "gpu_id": 0,
    "wandb_run_name": args.wandb_run_name ### FILL YOUR NAME HERE
    }

    wandb.init(entity = "manugaur", project = "clip_cap_reproduce", config = args)
    # wandb.run.name = config["wandb_run_name"]
    wandb.run.name = config["wandb_run_name"]


class Summary():
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.val_history = list()
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.val_history = list()

    def update(self, val, n=1):
      #n : batch size
      #val :avg loss
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.val_history.append(val) # maintaining a list of val losses.

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)
        return fmtstr.format(**self.__dict__)

class CocoDataset(Dataset):
    """
    inputs : data dict--> RN50 clip embeddings, corresponding captions.
    returns : clip embedding, tokenzied caption, mask (over prefix,tokens and padding)
    """
    
    def __init__(self, data_path, prefix_len,norm_prefix, tokenizer = "gpt2"):
        self.data = open_pickle(data_path)
        self.clip_embed = self.data['clip_embedding']
        self.meta_data = self.data['captions']
        self.tokenizer = GPT2Tokenizer.from_pretrained(tokenizer)
        self.prefix_len = prefix_len
        self.norm_prefix = norm_prefix
        self.split = data_path.split("_")[-1][:-4]
                
        #dataset needs to be arranged so a given 'idx' --> clip_embed of image, tokenized caption.
        # cannot tokenize everytime. Too expensive.

        self.indexed_dataset_path = f"/ssd_scratch/cvit/manu/clip_cap_manu/{self.split}_caption_tokens.pkl"
        if os.path.isfile(self.indexed_dataset_path):
            print("loading data.... ")
            self.tokenized_captions, self.max_len_token = open_pickle(self.indexed_dataset_path)
        else:
            #using a given idx, we can access the clip embedding and its corresponding tokenized caption 
            print("creating data")
            self.tokenized_captions = []
            token_len_list = []

            for meta_data in self.meta_data:
                tokens = torch.tensor(self.tokenizer.encode(meta_data['caption']),dtype=torch.int)
                self.tokenized_captions.append(tokens)
                token_len_list.append(tokens.shape[-1])
            
            all_len = torch.tensor(token_len_list, dtype = torch.float)
            #max = 182
            
            self.max_len_token = min(all_len.mean() + 10*(all_len.std()), all_len.max())

            dump_pickle((self.tokenized_captions, self.max_len_token), self.indexed_dataset_path)
        # # which clip embedding to be returned along with a given caption
        # self.caption2clip_idx = [x['clip_embedding'] for x in self.meta_data]

    def __len__(self):
        return len(self.data['clip_embedding'])
        # return 1200

        
    def pad(self, idx):
        tokens = self.tokenized_captions[idx]
        padding = int(self.max_len_token - tokens.shape[-1])
        if padding>0:
            pad = torch.zeros(padding)
            pad = pad.masked_fill(pad ==0, -1)
            tokens = torch.cat((tokens, pad)).int()

            ### padded tokens replace the tokens. Here the padding is done by -1. But the tokens returned by the method have padding with 0.
            self.tokenized_captions[idx] = tokens
        else:
            tokens = tokens[:int(self.max_len_token)]
            self.tokenized_captions[idx] = tokens
        mask = tokens.ge(0)
        tokens[~mask] =0
        mask = torch.cat((torch.ones(self.prefix_len),mask))
        
        return tokens, mask


    def __getitem__(self, idx): 
            
        padded_tokens, mask = self.pad(idx)
        prefix = self.clip_embed[idx]
        if self.norm_prefix:
            prefix = prefix.float()
            prefix = prefix/prefix.norm(2,-1) # L2 norm along the last dimension
        
        return prefix, padded_tokens, mask


class MlpTransformer(nn.Module):
    def __init__(self, in_dim, h_dim, out_d: Optional[int] = None, act=nnf.relu, dropout=0.):
        super().__init__()
        out_d = out_d if out_d is not None else in_dim
        self.fc1 = nn.Linear(in_dim, h_dim)
        self.act = act
        self.fc2 = nn.Linear(h_dim, out_d)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class MultiHeadAttention(nn.Module):

    def __init__(self, dim_self, dim_ref, num_heads, bias=True, dropout=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim_self // num_heads
        self.scale = head_dim ** -0.5
        self.to_queries = nn.Linear(dim_self, dim_self, bias=bias)
        self.to_keys_values = nn.Linear(dim_ref, dim_self * 2, bias=bias)
        self.project = nn.Linear(dim_self, dim_self)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, y=None, mask=None):
        y = y if y is not None else x
        b, n, c = x.shape
        _, m, d = y.shape
        # b n h dh
        queries = self.to_queries(x).reshape(b, n, self.num_heads, c // self.num_heads)
        # b m 2 h dh
        keys_values = self.to_keys_values(y).reshape(b, m, 2, self.num_heads, c // self.num_heads)
        keys, values = keys_values[:, :, 0], keys_values[:, :, 1]
        attention = torch.einsum('bnhd,bmhd->bnmh', queries, keys) * self.scale
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1)
            attention = attention.masked_fill(mask.unsqueeze(3), float("-inf"))
        attention = attention.softmax(dim=2)
        out = torch.einsum('bnmh,bmhd->bnhd', attention, values).reshape(b, n, c)
        out = self.project(out)
        return out, attention


class TransformerLayer(nn.Module):

    def forward_with_attention(self, x, y=None, mask=None):
        x_, attention = self.attn(self.norm1(x), y, mask)
        x = x + x_
        x = x + self.mlp(self.norm2(x))
        return x, attention

    def forward(self, x, y=None, mask=None):
        x = x + self.attn(self.norm1(x), y, mask)[0]
        x = x + self.mlp(self.norm2(x))
        return x

    def __init__(self, dim_self, dim_ref, num_heads, mlp_ratio=4., bias=False, dropout=0., act=nnf.relu,
                 norm_layer: nn.Module = nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim_self)
        self.attn = MultiHeadAttention(dim_self, dim_ref, num_heads, bias=bias, dropout=dropout)
        self.norm2 = norm_layer(dim_self)
        self.mlp = MlpTransformer(dim_self, int(dim_self * mlp_ratio), act=act, dropout=dropout)


class Transformer(nn.Module):

    def forward_with_attention(self, x, y=None, mask=None):
        attentions = []
        for layer in self.layers:
            x, att = layer.forward_with_attention(x, y, mask)
            attentions.append(att)
        return x, attentions

    def forward(self, x, y=None, mask=None):
        for i, layer in enumerate(self.layers):
            if i % 2 == 0 and self.enc_dec: # cross
                x = layer(x, y)
            elif self.enc_dec:  # self
                x = layer(x, x, mask)
            else:  # self or cross
                x = layer(x, y, mask)
        return x

    def __init__(self, dim_self: int, num_heads: int, num_layers: int, dim_ref: Optional[int] = None,
                 mlp_ratio: float = 2., act=nnf.relu, norm_layer: nn.Module = nn.LayerNorm, enc_dec: bool = False):
        super(Transformer, self).__init__()
        dim_ref = dim_ref if dim_ref is not None else dim_self
        self.enc_dec = enc_dec
        if enc_dec:
            num_layers = num_layers * 2
        layers = []
        for i in range(num_layers):
            if i % 2 == 0 and enc_dec:  # cross
                layers.append(TransformerLayer(dim_self, dim_ref, num_heads, mlp_ratio, act=act, norm_layer=norm_layer))
            elif enc_dec:  # self
                layers.append(TransformerLayer(dim_self, dim_self, num_heads, mlp_ratio, act=act, norm_layer=norm_layer))
            else:  # self or cross
                layers.append(TransformerLayer(dim_self, dim_ref, num_heads, mlp_ratio, act=act, norm_layer=norm_layer))
        self.layers = nn.ModuleList(layers)


class Model(nn.Module):

    def __init__(self, clip_dim,prefix_len, const_len,num_layers,only_projection = False):
        super().__init__()
        self.clip_dim = clip_dim
        self.prefix_len = prefix_len
        self.const_len = const_len
        self.only_projection = only_projection
        self.gpt = GPT2LMHeadModel.from_pretrained('gpt2')
        self.gpt_dim = self.gpt.transformer.wte.weight.shape[1]     #token embedding weight.shape
        self.linear = nn.Linear(self.clip_dim,self.prefix_len*(self.gpt_dim))
        self.learnable_const = nn.Parameter(torch.randn(self.const_len, self.gpt_dim), requires_grad=True)
        self.transformer = Transformer(self.gpt_dim, 8, num_layers)  #token_embedding, attn_heads, num_blocks
        
    def forward(self, clip_embed, tokens = None, mask = None):
        #prefix --> linear layer --> transformer --> output + caption tokens --> gpt        
        
        # project clip embed to gpt space
        x = self.linear(clip_embed.to(torch.float32)).view(clip_embed.shape[0],self.prefix_len, -1) # (B,K,gpt_dim)
        #concat learnable constant and clip embedding mapped to gpt space
        learnable_const = self.learnable_const.unsqueeze(0).expand(clip_embed.shape[0],*self.learnable_const.shape) # (B,K,gpt_dim)
        x = torch.cat((x,learnable_const),dim = 1) # (B,2K,gpt_dim)

        # align the clip embedding to gpt space using a transformer. 
        x = self.transformer(x)[:,self.prefix_len:] #Extract only learnable constant
        
        if self.only_projection:
            return x 
        
        # feed the gpt (learnable constant + tokenized_caption)
        token_embed = self.gpt.transformer.wte(tokens)    # (vocab_size, token_embedding=768) --> (B,T, 768)        
        x = torch.cat((x,token_embed),dim = 1)

        out = self.gpt(inputs_embeds = x, attention_mask = mask)

        return out

# def save_config(args: argparse.Namespace):
#     config = {}
#     for key, item in args._get_kwargs():
#         config[key] = item
#     out_path = os.path.join(args.out_dir, f"{args.prefix}.json")
#     with open(out_path, 'w') as outfile:
#         json.dump(config, outfile)


@torch.no_grad()
def evaluate(model, val_dataloader,val_dataset, device):
    model.eval()
    loss_meter = AverageMeter("eval_loss", ":.5f")
    for prefix, tokens, mask in val_dataloader:
        tokens, mask, prefix = tokens.to(device), mask.to(device), prefix.to(device, dtype=torch.float32)
        #get the class idx for each instance in the batch.
        outputs = model(prefix,tokens, mask)
        logits = outputs.logits[:, val_dataset.prefix_len - 1: -1]
        loss = nnf.cross_entropy(logits.reshape(-1, logits.shape[-1]), tokens.to(torch.long).flatten(), ignore_index=0)
        loss_meter.update(loss.item(), tokens.shape[0])
        return loss_meter

def load_model(config_path: str, epoch_or_latest: Union[str, int] = '_latest'):
    with open(config_path) as f:
        config = json.load(f)
    parser = argparse.ArgumentParser()
    parser.set_defaults(**config)
    args = parser.parse_args()
    if type(epoch_or_latest) is int:
        epoch_or_latest = f"-{epoch_or_latest:03d}"
    model_path = os.path.join(args.out_dir, f"{args.prefix}{epoch_or_latest}.pt")
    if args.only_prefix:
        model = ClipCaptionPrefix(args.prefix_length)
    else:
        model = ClipCaptionModel(args.prefix_length)
    if os.path.isfile(model_path):
        print(f"loading model from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    else:
        print(f"{model_path} is not exist")
    return model, parser


def train(train_dataset, val_dataset, model, config, warmup_steps= 5000, output_dir = "."):
    val_min = float(1000)
    device = torch.device('cuda:0') if torch.cuda.is_available()  else torch.device('cpu')
    batch_size = config['batch']
    epochs = config['num_epochs']
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model = model.to(device)
    model.train()
    loss_meter = AverageMeter("train_loss", ":.5f")
    optimizer = AdamW(model.parameters(), lr=float(config['lr']))
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=5000, num_training_steps=epochs* len(train_dataloader))
    
    step = 0
    for epoch in range(epochs):
        
        print(f">>> Training epoch {epoch}")
        sys.stdout.flush()

        # progress = tqdm(total=len(train_dataloader), desc=output_prefix)
        epoch_train_losses = []
        for idx, (prefix, tokens, mask) in enumerate(train_dataloader):
            step+=1
            model.zero_grad()
            optimizer.zero_grad()
            tokens, mask, prefix = tokens.to(device), mask.to(device), prefix.to(device, dtype=torch.float32)
            outputs = model(prefix,tokens, mask)
            logits = outputs.logits[:, train_dataset.prefix_len - 1: -1]
            loss = nnf.cross_entropy(logits.reshape(-1, logits.shape[-1]), tokens.to(torch.long).flatten(), ignore_index=0)
            epoch_train_losses.append(loss.item())
            loss_meter.update(loss.item(), tokens.shape[0])
            loss.backward()
            optimizer.step()
            scheduler.step()
            #log info step-wise
            train_log = {"epoch": epoch+1,
            "train_loss_avg": loss_meter.avg,
            "lr": optimizer.state_dict()["param_groups"][0]["lr"],}
            if use_log:
                wandb.log(train_log, step = step)

        #Calculate validation loss 
        val_loss_meter = evaluate(model, val_dataloader, val_dataset, device)
        val_log = {
            "val_loss_avg": val_loss_meter.avg,
            }
        if use_log: 
            wandb.log(val_log, step = step)
            logging.info({**train_log, **val_log})
        # progress.set_postfix({"loss": loss.item()})
        # progress.update()
    # progress.close()
        # print(data_to_log)
        if save_ckpt:
                if val_loss_meter.avg< val_min:
                    val_min = val_loss_meter.avg
                    torch.save(
                        model.state_dict(),
                        os.path.join(output_dir, f"coco_layers_{config['num_layers']}.pt"),
                    )
    return model


def trigger_training(config):

    train_dataset = CocoDataset(config['train_data'], config['prefix_length'],config['normalize_prefix'])
    val_dataset = CocoDataset(config['val_data'], config['prefix_length'],config['normalize_prefix'])
    model = Model(clip_dim = config['prefix_dim'], prefix_len = config['prefix_length'], const_len =config['prefix_length'], num_layers = config['num_layers'])
    train(train_dataset,val_dataset,model, config, output_dir=config['out_dir'])

def sweep_agent_manager():
    wandb.init()
    config = dict(wandb.config)
    run_name = f"num_layers_{config['num_layers']}_prefix_len_10"
    wandb.run.name = run_name
    logging.basicConfig(filename=f'/home2/manugaur/clip_cap_manu/logs/num_layer_sweep/{run_name}.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    trigger_training(config)

def main():
    # with open('/home2/manugaur/clip_cap_manu/sweep_config.yml') as f:
    #     config = yaml.load(f,Loader=yaml.FullLoader)
    # trigger_training(config)
    wandb.agent(sweep_id="manugaur/clip_cap_reproduce/1mth7imu", function=sweep_agent_manager, count=100)

if __name__ == '__main__':
    main()

"""

 Why self.caption_tokens[idx] = tokens ::::: here padding is -1

"""

