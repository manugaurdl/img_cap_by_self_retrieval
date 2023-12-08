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

class CocoDataset(Dataset):
    """
    inputs : data dict-->  clip embeddings,  corresponding captions.
    returns : clip embedding, tokenzied caption, mask.
              Caption tokens are padded. Max length of 40 is ensured. 
              Mask is of length 50 however. As it has torch.ones(1,10) prepended to captions mask for image prefix embedding. 
    """
    
    def __init__(self, data_path, prefix_len,norm_prefix, split, tokenizer):
        """
        data_path : {model}_{split}_emb.pkl 
        indexed_dataset_path : {split}_cap2tion_tokens
        """
        self.data = open_pickle(data_path)
        self.clip_embed = self.data['clip_embedding']
        self.captions = self.data['caption']
        self.cocoids = self.data['cocoid']
        self.filenames = self.data['filename']
        self.sent_ids = self.data['sent_id']
        

        self.tokenizer = GPT2Tokenizer.from_pretrained(tokenizer)
        self.prefix_len = prefix_len
        self.norm_prefix = norm_prefix
        self.split = split

        #dataset needs to be arranged so a given 'idx' --> clip_embed of image, tokenized caption.
        # cannot tokenize everytime. Too expensive.
        
        self.indexed_dataset_path = os.path.join(data_path.split('/img_cap_self_retrieval_clip/data')[0],f'img_cap_self_retrieval_clip/data/{self.split}_caption_tokens.pkl')
        if os.path.isfile(self.indexed_dataset_path):
            print(f"loading {self.split} data.... ")
            self.tokenized_captions, self.max_len_token = open_pickle(self.indexed_dataset_path)
        else:
            #using a given idx, we can access the clip embedding and its corresponding tokenized caption 
            print(f"tokenizing {self.split} captions")
            self.tokenized_captions = []
            token_len_list = []

            for caption in self.captions:
                
                tokens = torch.tensor(self.tokenizer.encode(caption),dtype=torch.int)
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
        # return 400
        
    def pad(self, idx):
        #### this is the problem. I need diff tokenized caption.
        tokens = self.tokenized_captions[idx]
        padding = round(float(self.max_len_token)) - tokens.shape[-1]
        # padding = int(self.max_len_token - tokens.shape[-1])

        if padding>0:
            pad = torch.zeros(padding)

            pad = pad.masked_fill(pad ==0, -1)
            tokens = torch.cat((tokens, pad)).int()

            ### padded tokens replace the tokens. Here the padding is done by -1. But the tokens returned by the method have padding with 0.
            self.tokenized_captions[idx] = tokens
        else:
            tokens = tokens[:round(float(self.max_len_token))]
            self.tokenized_captions[idx] = tokens
        mask = tokens.ge(0)
        tokens[~mask] =0
        mask = torch.cat((torch.ones(self.prefix_len),mask))
        
        return tokens, mask


    def __getitem__(self, idx): 
        
        padded_tokens, mask = self.pad(idx)
        prefix = self.clip_embed[idx]
        # meta data
        cocoid = self.cocoids[idx] 
        filename = self.filenames[idx]
        sent_id = self.sent_ids[idx]

        if self.norm_prefix:
            prefix = prefix.float()
            prefix = prefix/prefix.norm(2,-1) # L2 norm along the last dimension
        
        return prefix, padded_tokens, mask, {"cocoid" : cocoid, "filename" : filename, "sent_id" : sent_id}
