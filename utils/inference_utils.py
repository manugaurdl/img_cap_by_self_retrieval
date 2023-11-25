import clip
import os
from torch import nn
import numpy as np
import torch
from enum import Enum
import random
import json
from tqdm import tqdm, trange
import climage
import torch.nn.functional as nnf
import sys
from typing import Tuple, List, Union, Optional
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    AdamW,
    get_linear_schedule_with_warmup,
)
import skimage.io as io
import PIL.Image



def generate_beam(model, tokenizer, beam_size: int = 5, prompt=None, embed=None,
                  entry_length=67, temperature=1., stop_token: str = '.'):

    model.eval()
    stop_token_index = tokenizer.encode(stop_token)[0]
    tokens = None
    scores = None
    device = next(model.parameters()).device
    seq_lengths = torch.ones(beam_size, device=device)
    is_stopped = torch.zeros(beam_size, device=device, dtype=torch.bool)
    with torch.no_grad():
        if embed is not None:
            generated = embed
        else:
            if tokens is None:
                tokens = torch.tensor(tokenizer.encode(prompt))
                tokens = tokens.unsqueeze(0).to(device)
                generated = model.gpt.transformer.wte(tokens)
        for i in range(entry_length):
            outputs = model.gpt(inputs_embeds=generated)
            logits = outputs.logits
            logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
            logits = logits.softmax(-1).log()
            if scores is None:
                scores, next_tokens = logits.topk(beam_size, -1)
                generated = generated.expand(beam_size, *generated.shape[1:])
                next_tokens, scores = next_tokens.permute(1, 0), scores.squeeze(0)
                if tokens is None:
                    tokens = next_tokens
                else:
                    tokens = tokens.expand(beam_size, *tokens.shape[1:])
                    tokens = torch.cat((tokens, next_tokens), dim=1)
            else:
                logits[is_stopped] = -float(np.inf)
                logits[is_stopped, 0] = 0
                scores_sum = scores[:, None] + logits
                seq_lengths[~is_stopped] += 1
                scores_sum_average = scores_sum / seq_lengths[:, None]
                scores_sum_average, next_tokens = scores_sum_average.view(-1).topk(beam_size, -1)
                next_tokens_source = next_tokens // scores_sum.shape[1]
                seq_lengths = seq_lengths[next_tokens_source]
                next_tokens = next_tokens % scores_sum.shape[1]
                next_tokens = next_tokens.unsqueeze(1)
                tokens = tokens[next_tokens_source]
                tokens = torch.cat((tokens, next_tokens), dim=1)
                generated = generated[next_tokens_source]
                scores = scores_sum_average * seq_lengths
                is_stopped = is_stopped[next_tokens_source]
            next_token_embed = model.gpt.transformer.wte(next_tokens.squeeze()).view(generated.shape[0], 1, -1)
            generated = torch.cat((generated, next_token_embed), dim=1)
            is_stopped = is_stopped + next_tokens.eq(stop_token_index).squeeze()
            if is_stopped.all():
                break
    scores = scores / seq_lengths
    output_list = tokens.cpu().numpy()
    output_texts = [tokenizer.decode(output[:int(length)]) for output, length in zip(output_list, seq_lengths)]
    order = scores.argsort(descending=True)
    output_texts = [output_texts[i] for i in order]
    return output_texts


def generate2(
        model,
        tokenizer,
        tokens=None,
        prompt=None,
        embed=None,
        entry_count=1,
        entry_length=67,  # maximum number of words
        top_p=0.8,
        temperature=1.,
        stop_token: str = '.',
):
    model.eval()
    generated_num = 0
    generated_list = []
    stop_token_index = tokenizer.encode(stop_token)[0]
    filter_value = -float("Inf")
    device = next(model.parameters()).device # (10,768)

    with torch.no_grad():

        for entry_idx in trange(entry_count):
            if embed is not None:
                # generated embeddings are concatenated.
                # at start = (B, 10, 768)
                generated = embed  
            else:
                if tokens is None:
                    tokens = torch.tensor(tokenizer.encode(prompt))
                    tokens = tokens.unsqueeze(0).to(device)

                generated = model.gpt.transformer.wte(tokens)

            # each iteration, the context on which generation is conditioned increases by 1. (prefix + gpt_outputs)
            for i in range(entry_length):

                outputs = model.gpt(inputs_embeds=generated)
                # logits for next token is a distribution over vocab_size. Can sample a token from it and decode it into a word. 
                logits = outputs.logits

                # Using the output of time step and concatenating it with previous outputs. this (generated) is fed to gpt at next time step. 

                #we consider the logit of the last token
                # the logit of last token has the encoded context of all the previous tokens.
                logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)  # (B, T, vocab_size)
               
               # ----------------------------
                # """
                # TAKING THE LARGEST LOGIT
                # next_token = torch.argmax(logits, dim = -1).unsqueeze(0)

                # """
                # ----------------------------
               
                # """                
                # torch.multinomial

                probs = torch.nn.functional.softmax(logits, dim=-1) # (B, C)
                # sample from the distribution
                next_token = torch.multinomial(probs, num_samples=1) # (B, 1)
                
                # """
               # ---------------------------- 
                """
                THEIR IMPLEMENTATION
                # sort logits in descending order.
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                # take softmax across all the vocab_tokens --> They all sum to one and are arranged in descendng order
                # take cumulative sum. [0.4, 0.3,0.2,0.1] --> [0.4,0.7,0.9,1]
                cumulative_probs = torch.cumsum(nnf.softmax(sorted_logits, dim=-1), dim=-1)
                # The first few vocab tokens have the highest probability. For ex: out of 50k tokens, first 4 might comprise of .8 cumulative prob.
                # we retain only those tokens
                sorted_indices_to_remove = cumulative_probs > top_p
                # currently, don't drop first k indices
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                                                    ..., :-1
                                                    ].clone()
                # #now, don't drop first k+1 elements
                sorted_indices_to_remove[..., 0] = 0  # isn't this redundant??
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                # The logits has a score for each vocab token. For indices_to_remove, that score = -inf
                # while taking softmax, those indices are ignored
                logits[:, indices_to_remove] = filter_value
                # Find idx (vocab token) for which the logit score is highest.
                
                next_token = torch.argmax(logits, -1).unsqueeze(0) # idx for a vocab token (B, 1)
                """
                # ----------------------------

                # for the given idx, go the nn.Embedding table and get token embedding for this idx
                next_token_embed = model.gpt.transformer.wte(next_token) # (B,1,768)
                if tokens is None:
                    tokens = next_token
                else:
                    tokens = torch.cat((tokens, next_token), dim=1)
                #the generated embeddings are increased by 1 at each iteration. 
                generated = torch.cat((generated, next_token_embed), dim=1)
                if stop_token_index == next_token.item():
                    break

            output_list = list(tokens.squeeze().cpu().numpy())
            output_text = tokenizer.decode(output_list)
            generated_list.append(output_text)

    return generated_list[0]

