import torch
import torch.nn as nn
from torch.nn import functional as F


def Reinforce(policy_logprob, policy_cap, reward, reduction='mean'):
    """
    policy_logprob : (B * sample_n, 40) --> logprob for each word for each policy caption for each image.
    policy_cap : sampled captions [B * sample_n, 40]
    
    For an caption i : 
        policy_cap[i] : token
        policy_logprob[i] : logprob for that token

    reward : R(c,I) - b    [B * sample_n, 40] (same reward for each word in a given cap.)
    """
    N,L = policy_logprob.shape[:2]
    # I have already gathered logprobs for sampled words.
    # input = input.gather(2, seq.unsqueeze(2)).squeeze(2)
    
    # flatten logprobs and rewards for all words
    policy_logprob = policy_logprob.reshape(-1)
    reward = reward.reshape(-1)
    
    #mask the padded words as False
    mask = (policy_cap.data>0).to(policy_logprob)

    #Flatten the mask. Padded words = 0
    
    # mask = torch.cat([mask.new(mask.size(0), 1).fill_(1), mask[:, :-1]], 1).reshape(-1)
    mask = mask.reshape(-1)
    
    # padded word --> words sampled after stop token. 
    # loss only calculated for policy caption i.e until first stop token.
    output = - policy_logprob * reward * mask
    
    if reduction == 'none':
        output = output.view(N,L).sum(1) / mask.view(N,L).sum(1)
    elif reduction == 'mean':
        output = torch.sum(output) / torch.sum(mask) #avg over entire batch.

    return output
