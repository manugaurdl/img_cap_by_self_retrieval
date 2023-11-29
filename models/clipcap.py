from .transformer import Transformer
import torch
import torch.nn as nn
from torch.nn import functional as nnf
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup

class Model(nn.Module):

    def __init__(self, clip_dim,prefix_len, const_len,num_layers):
        super().__init__()
        self.clip_dim = clip_dim
        self.prefix_len = prefix_len
        self.const_len = const_len
        self.gpt = GPT2LMHeadModel.from_pretrained('gpt2')
        self.gpt_dim = self.gpt.transformer.wte.weight.shape[1]     #token embedding weight.shape
        self.linear = nn.Linear(self.clip_dim,self.prefix_len*(self.gpt_dim))
        self.learnable_const = nn.Parameter(torch.randn(self.const_len, self.gpt_dim), requires_grad=True)
        self.transformer = Transformer(self.gpt_dim, 8, num_layers)  #token_embedding, attn_heads, num_blocks
        
    def forward(self, clip_embed, tokens = None, mask = None, only_prefix = False):
        #prefix --> linear layer --> transformer --> output + caption tokens --> gpt        
        
        # project clip embed to gpt space
        x = self.linear(clip_embed.to(torch.float32)).view(clip_embed.shape[0],self.prefix_len, -1) # (B,K,gpt_dim)
        #concat learnable constant and clip embedding mapped to gpt space
        learnable_const = self.learnable_const.unsqueeze(0).expand(clip_embed.shape[0],*self.learnable_const.shape) # (B,K,gpt_dim)
        x = torch.cat((x,learnable_const),dim = 1) # (B,2K,gpt_dim)

        # align the clip embedding to gpt space using a transformer. 
        learnable_const = self.transformer(x)[:,self.prefix_len:] #Extract only learnable constant
        
        if only_prefix:
            return learnable_const
        # feed the gpt (learnable constant + tokenized_caption)
        token_embed = self.gpt.transformer.wte(tokens)    # (vocab_size, token_embedding=768) --> (B,T, 768)        
        gpt_input = torch.cat((learnable_const,token_embed),dim = 1)

        out = self.gpt(inputs_embeds = gpt_input, attention_mask = mask)

        return out
