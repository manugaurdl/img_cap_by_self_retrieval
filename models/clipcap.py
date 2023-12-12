from .transformer import Transformer
import torch
import torch.nn as nn
from torch.nn import functional as nnf
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
from typing import Tuple, Optional, Union
from utils.helper_functions import *

class TransformerMapper(nn.Module):
    
    def __init__(self, prefix_len, clip_dim, gpt_dim,const_len, attn_heads, num_layers):
        super(TransformerMapper, self).__init__()

        self.learnable_const = nn.Parameter(torch.randn(const_len, gpt_dim), requires_grad=True)
        self.transformer = Transformer(gpt_dim, attn_heads, num_layers)  #token_embedding, attn_heads, num_blocks
        self.linear = nn.Linear(clip_dim,prefix_len*(gpt_dim))

        self.prefix_len = prefix_len

    def forward(self, x):
        # project clip embed to gpt space
        x = self.linear(x.to(torch.float32)).view(x.shape[0],self.prefix_len, -1) # (B,K,gpt_dim)
        
        #concat learnable constant and clip embedding mapped to gpt space
        learnable_const = self.learnable_const.unsqueeze(0).expand(x.shape[0],*self.learnable_const.shape) # (B,K,gpt_dim)

        x = torch.cat((x,learnable_const),dim = 1) # (B,2K,gpt_dim)
        learnable_const = self.transformer(x)[:,self.prefix_len:] #Extract only learnable constant
        
        return learnable_const

class MLP(nn.Module):

    def __init__(self, sizes: Tuple[int, ...], prefix_len, bias=True, act=nn.Tanh):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(act())
        self.model = nn.Sequential(*layers)
        self.prefix_len  = prefix_len
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        return x.view(x.shape[0],self.prefix_len,-1)


class Model(nn.Module):

    def __init__(self, clip_dim,prefix_len, const_len,attn_heads, num_layers,freeze_gpt,cocotalk):
        super().__init__()
        self.clip_dim = clip_dim
        self.prefix_len = prefix_len
        self.const_len = const_len
        self.attn_heads = attn_heads
        self.num_layers = num_layers
        self.freeze_gpt = freeze_gpt
        self.gpt = GPT2LMHeadModel.from_pretrained('gpt2')
        self.gpt_dim = self.gpt.transformer.wte.weight.shape[1]     #token embedding weight.shape

        # cocotalk vocab
        # self.vocab = open_json(cocotalk)['ix_to_word']
        # bad_endings = ['a','an','the','in','for','at','of','with','before','after','on','upon','near','to','is','are','am','the']
        # self.bad_endings_ix = [int(k) for k,v in self.vocab.items() if v in bad_endings]


        if self.freeze_gpt:
            self.mapping_network = TransformerMapper(self.prefix_len, self.clip_dim, self.gpt_dim, self.const_len, self.attn_heads, self.num_layers)
        else:
            self.mapping_network = MLP((self.clip_dim, (self.gpt_dim * self.prefix_len) // 2,
                                self.gpt_dim * self.prefix_len),self.prefix_len)   
        
    def forward(self, clip_embed, tokens = None, mask = None, only_prefix = False):
        #prefix --> linear layer --> transformer --> output + caption tokens --> gpt        
        
        prefix_proj = self.mapping_network(clip_embed)
        if only_prefix:
            return prefix_proj
        # embed tokenized caption to gpt dim
        token_embed = self.gpt.transformer.wte(tokens)    # (vocab_size, token_embedding=768) --> (B,T, 768)        
        gpt_input = torch.cat((prefix_proj,token_embed),dim = 1)

        out = self.gpt(inputs_embeds = gpt_input, attention_mask = mask)

        return out
    
    def parameters(self, recurse: bool = True):
        if self.freeze_gpt:
            return self.mapping_network.parameters()


        return super().parameters()

# model = Model()
# def surgery(model: Model):
    # for layer in model.layers():
    #   if isinstance(layer, LayerNorm):
            # layer.requires_grad = False

    # return model