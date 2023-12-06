import torch
import torch.nn as nn
from torch.nn import functional as F


def Reinforce(input, seq, reward, reduction='mean'):
    """
    policy_logprob : sample_logprobs [10, 20, 9488]
    seq : sampled captions [10,20]
    reward : R(c,I) - b    [10,20] (same value for each timestep.)
    """
    
    N,L = input.shape[:2]
    # I have already gathered logprobs for sampled words.
    # input = input.gather(2, seq.unsqueeze(2)).squeeze(2)
    
    # flatten logprobs and rewards for all words
    input = input.reshape(-1)
    reward = reward.reshape(-1)
    
    #mask for non padded words
    mask = (seq>0).to(input)
    mask = torch.cat([mask.new(mask.size(0), 1).fill_(1), mask[:, :-1]], 1).reshape(-1)
    output = - input * reward * mask # why mask? padded tokens will have 0 log prob anyways.

    if reduction == 'none':
        output = output.view(N,L).sum(1) / mask.view(N,L).sum(1)
    elif reduction == 'mean':
        output = torch.sum(output) / torch.sum(mask) #avg over entire batch.

    return output
