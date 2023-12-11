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
        import ipdb;ipdb.set_trace()

        self.tokenizer = GPT2Tokenizer.from_pretrained(tokenizer)
        self.prefix_len = prefix_len
        self.norm_prefix = norm_prefix
        self.split = split

        #dataset needs to be arranged so a given 'idx' --> clip_embed of image, tokenized caption.
        # cannot tokenize everytime. Too expensive.
        self.indexed_dataset_path = os.path.join(data_path.split('/new')[0],f'new_{self.split}_caption_tokens.pkl')
        if os.path.isfile(self.indexed_dataset_path):
            print(f"loading {self.split} data.... ")
            self.tokenized_captions, self.max_len_token = open_pickle(self.indexed_dataset_path)
            print(True)
        else:
            #using a given idx, we can access the clip embedding and its corresponding tokenized caption 
            print(f"tokenizing {self.split} captions")
            self.tokenized_captions = [] # list of list of captions 
            token_len_list = []

            for caption in tqdm(self.captions, total = len(self.captions)): 
                #caption : list of 5 cap for given cocoid
                tokens = [torch.tensor(self.tokenizer.encode(cap),dtype=torch.int) for cap in caption]
                self.tokenized_captions.append(tokens)
                token_len_list.extend([token.shape[-1] for token in tokens])
            
            all_len = torch.tensor(token_len_list, dtype = torch.float)
            #max = 182
            self.max_len_token = min(all_len.mean() + 10*(all_len.std()), all_len.max())

            dump_pickle((self.tokenized_captions, self.max_len_token), self.indexed_dataset_path)
        # # which clip embedding to be returned along with a given caption
        # self.caption2clip_idx = [x['clip_embedding'] for x in self.meta_data]

    def __len__(self):
        return len(self.data['clip_embedding'])
        # return 400
        
    def pad(self, idx, cap_idx, tokens):
                     
        padding = round(float(self.max_len_token)) - tokens.shape[-1]

        if padding>0:
            pad = torch.zeros(padding)

            pad = pad.masked_fill(pad ==0, -1)
            tokens = torch.cat((tokens, pad)).int()

            ### padded tokens replace the tokens. Here the padding is done by -1. But the tokens returned by the method have padding with 0.
            self.tokenized_captions[idx][cap_idx] = tokens
        else:
            # if caption > max_len, truncate it 
            tokens = tokens[:round(float(self.max_len_token))]
            self.tokenized_captions[idx][cap_idx] = tokens
            
        mask = tokens.ge(0)
        tokens[~mask] =0
        mask = torch.cat((torch.ones(self.prefix_len),mask))
        
        return (tokens, mask)


    def __getitem__(self, idx): 
        
        padded_tokens, mask =  zip(*[self.pad(idx, cap_idx, tokens) for cap_idx, tokens in enumerate(self.tokenized_captions[idx])])   # list of 5 tokenized captions
        padded_tokens = torch.stack((padded_tokens),dim=0)
        mask = torch.stack((mask),dim=0)
        prefix = self.clip_embed[idx]
        # meta data
        cocoid = self.cocoids[idx] 
        filename = self.filenames[idx]
        sent_id = self.sent_ids[idx]
        untokenized_cap = self.captions[idx]

        if self.norm_prefix:
            prefix = prefix.float()
            prefix = prefix/prefix.norm(2,-1) # L2 norm along the last dimension
        
        return prefix, padded_tokens, mask, untokenized_cap, {"cocoid" : cocoid, "filename" : filename, "sent_id" : sent_id}


# # return (x,y1)....(x,y2)
# def mle_collate(batch):
#     """
#     batch = list of bsz
#     batch[0] = tuple of len 5

    
#     """
#     #(x,[y1,y2,y3]) --> (x,y1)...(x,y2)

#     new_batch = [] #add items to it

#     for i in range(batch[0][1].shape[0]):
        
#         x = list(batch[0]).copy()  # batch item created. 
        
#         #Everything preserved except tokens, mask, sent_id in meta_data.

#         x[1] = x[1][i,:] #tokens
#         x[2] = x[2][i,:] # mask
#         copy_dict = x[4].copy()
#         x[4]['sent_id']= copy_dict['sent_id'][i] # dict
#         new_batch.append(x)

#     print(True)
    
#     return batch
