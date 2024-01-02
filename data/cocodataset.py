import json
import os
import numpy as np
import random
import torch
from torch.utils.data import Dataset, DataLoader
from utils.helper_functions import open_pickle, dump_pickle, save_model
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
from models.clipcap import Model
from models.transformer import *
from tqdm import tqdm
import pickle


class CocoDataset(Dataset):
    """
    inputs : data dict-->  clip embeddings,  cocoid.
    returns : clip embedding, tokenzied caption, mask.
              Caption tokens are padded. Max length of 40 is ensured. 
              Mask is of length 50 however. As it has torch.ones(1,10) prepended to captions mask for image prefix embedding. 
    """
    
    def __init__(self, data_path, prefix_len,norm_prefix, split, tokenizer, lazy_load):

        """
        data_path : {model}_{split}_emb.pkl , cocoid2caption.pkl
        indexed_dataset_path : {split}_caption_tokens
        """
        self.lazy_load = lazy_load
        self.data = open_pickle(data_path)
        self.clip_embed = self.data['clip_embedding']
        self.images = self.data['images']  # list of instances (images) for given split. Each is a dict of {cocoid, clip_idx}
        self.tokenizer = GPT2Tokenizer.from_pretrained(tokenizer)
        self.prefix_len = prefix_len
        self.norm_prefix = norm_prefix
        self.split = split
        self.max_len_token = 40  #based on train set
        self.id2cap = open_pickle(os.path.join(data_path.split('ViT')[0],"cocoid2caption.pkl"))

        # cannot tokenize everytime. Too expensive.
        self.indexed_dataset_path = os.path.join(data_path.split('ViT')[0],f'{self.split}_caption_tokens.pkl')
        
        if os.path.isfile(self.indexed_dataset_path):
            print(f"loading {self.split} data.... ")
            
            if not self.lazy_load:
                self.tokenized_captions, _ = open_pickle(self.indexed_dataset_path) # max_len used from train set.
            
            self.cocoid2tokenidx = open_pickle(os.path.join(data_path.split('ViT')[0],f'cocoid2tokenidx_{self.split}.pkl'))
        else:
            #using a given idx, we can access the clip embedding and its corresponding tokenized caption 
            print(f"tokenizing {self.split} captions")

            #iterate cocoids in self.images. get caption using id2cap. tokenize it. 
            #in __getitem__  : use idx --> images --> cocoid --> token_idx --> tokenzied cap 
            
            self.tokenized_captions = [] # list of list of captions 
            
            token_len_list = []
            self.cocoid2tokenidx = {}  #  cocoid --> idx for self.tokenized_captions

            for idx, image in enumerate(self.images): 
                
                cocoid = image['cocoid']
                caption = self.id2cap[cocoid]

                #caption : list of 5 cap for given cocoid
                tokens = [torch.tensor(self.tokenizer.encode(cap),dtype=torch.int) for cap in caption]
                
                self.tokenized_captions.append(tokens)
                
                token_len_list.extend([token.shape[-1] for token in tokens]) # sotre list of lengths of all tokens.

                self.cocoid2tokenidx[cocoid] = idx

            all_len = torch.tensor(token_len_list, dtype = torch.float)
            #max = 182
            self.max_len_token = min(all_len.mean() + 10*(all_len.std()), all_len.max())
            
            dump_pickle(self.cocoid2tokenidx,os.path.join(data_path.split('ViT')[0],f'cocoid2tokenidx_{self.split}.pkl'))
            dump_pickle((self.tokenized_captions, self.max_len_token), self.indexed_dataset_path)

    def __len__(self):
        return len(self.data['clip_embedding'])
        
    def pad(self, idx, cap_idx, tokens):
                     
        padding = self.max_len_token - tokens.shape[-1]

        if padding>0:
            pad = torch.zeros(padding)
            pad = pad.masked_fill(pad ==0, -1) 
            tokens = torch.cat((tokens, pad)).int() # tokens is padded with -1.

            ### padded tokens replace the tokens. Here the padding is done by -1. But the tokens returned by the method have padding with 0.
            if not self.lazy_load:
                self.tokenized_captions[idx][cap_idx] = tokens
        else:
            # if caption > max_len, truncate it 
            tokens = tokens[:self.max_len_token]
            if not self.lazy_load:
                self.tokenized_captions[idx][cap_idx] = tokens
            
        mask = tokens.ge(0) #True for indices > 0 i,e padded indices = False
        tokens[~mask] =0  # padding now done with 0
        mask = torch.cat((torch.ones(self.prefix_len),mask)) 
        
        return (tokens, mask)


    def __getitem__(self, idx):
        image = self.images[idx]
        cocoid = image['cocoid']
        prefix = self.clip_embed[image['clip_embedding']]  # shape : [512]
        token_idx = self.cocoid2tokenidx[cocoid]
        if self.lazy_load:
            with open(f"/ssd_scratch/cvit/manu/img_cap_self_retrieval_clip/data/lazy_load_train/{token_idx}.pkl", "rb") as f:
                unpadded_tokens = pickle.load(f)[:5]
        else:
            unpadded_tokens = self.tokenized_captions[token_idx][:5]

        padded_tokens, mask =  zip(*[self.pad(token_idx, cap_idx, tokens) for cap_idx, tokens in enumerate(unpadded_tokens)])   # list of 5 tokenized captions
        padded_tokens = torch.stack((padded_tokens),dim=0) #shape : [5,40]
        mask = torch.stack((mask),dim=0)
        # meta data
        untokenized_cap = self.id2cap[cocoid][:5]

        # if self.norm_prefix:
        #     prefix = prefix.float()
        #     prefix = prefix/prefix.norm(2,-1) # L2 norm along the last dimension
        
        return prefix, padded_tokens, mask, untokenized_cap, cocoid