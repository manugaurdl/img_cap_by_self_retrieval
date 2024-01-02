import torch
import torch.nn as nn
from torch.nn import functional as F
from utils.train_algos import *
from utils.rewards import *
from utils.eval_utils import *
from utils.losses import * 


def SCST(model,prefix, targets, mask,max_length, stop_token, tokenizer, config):
    
    model.eval()
    with torch.no_grad():
        prefix_embed = model(prefix, targets, mask, only_prefix = True)
        # greedy caption used only for self critical reward

        # T1 = time.time()
        _, _, greedy_cap  = sample(max_length, prefix_embed, model, config['temp'], "greedy", stop_token,tokenizer, config)

        # _, policy_seqLogprob, policy_cap = sample(max_length, prefix_embed, model, config['temp'], "greedy", stop_token,tokenizer, config)
        # step_time_avg.append(time.time() - T1)
        # print(len(step_time_avg))
        # print(f"bsz {config['batch_size']} sample_n {config['train_sample_n']} : {np.mean(np.array(step_time_avg))}")

    #currently overriding train() method
    model.train()
    prefix_embed = model(prefix, targets, mask, only_prefix = True)

    #trainable policy 
    # T1 = time.time()
    _, policy_seqLogprob, policy_cap = sample(max_length, prefix_embed, model, config['temp'], "sample", stop_token, tokenizer,config, sample_n = config['train_sample_n'])  # don't need logits (dist over all words). Have log prob for sampled word

    # step_time_avg.append(time.time() - T1)
    # print(len(step_time_avg))
    # print(f"sample_n {config['train_sample_n']} : {np.mean(np.array(step_time_avg))}")

    # len(gts) = bsz.
    # For each instance in batch --> 5 target cap.
    gts = tuple(np.split(np.array(targets.cpu()), prefix.shape[0]))
    # gts = torch.chunk(targets.cpu(), prefix.shape[0], dim=0)
    out = {}
    # R(c,I) -b : (sample_n* B, max_len)  --> each generated word in for policy cap 'i' gets same reward.
    # Per image, hence sample_n rewards.
    reward = get_self_critical_reward(greedy_cap, gts, policy_cap, config) #(n*B,40)
    # reward is moved to same device as logprobs for each sampled word for each caption.
    reward = torch.from_numpy(reward).to(policy_seqLogprob)
    loss = Reinforce(policy_seqLogprob, policy_cap.data, reward)
    # (R(c,I) -b) averaged over batch. For a given caption, reward is same for log_probs of all the words generated
    import ipdb;ipdb.set_trace()

    #reward is averaged over num policy captions
    return reward[:,0].mean(), loss

def LMCriterion(model, prefix, targets, mask, meta_data, prefix_len):
    
    outputs = model(prefix,targets, mask)

    # logits corresponding to preds for all caption tokens are taken.
    # i.e FROM logit of last learnable token TO logit of second last caption token.

    logits = outputs.logits[:, prefix_len - 1: -1]  #(B,41, vocab_size)
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.to(torch.long).flatten(), ignore_index=0) # (B,T) flattened to (B*T)

    probs = torch.nn.functional.softmax(logits, dim=-1)
    preds = torch.multinomial(probs.view(-1, probs.shape[-1]), num_samples=1) # preds is flattened out --> (B*max_cap_len , 1)
    
    # epoch_train_preds.append(preds.view(batch_size,targets.shape[-1], -1).squeeze(-1)) # (B,41) --> each pred is same shape as targets obviously. 
    preds = preds.view(prefix.shape[0],targets.shape[-1], -1).squeeze(-1) # reshape to (B, max_cap_len)
    entropy = -(F.softmax(logits, dim=2) * logits).sum(2).sum(1) / ((preds>0).to(logits).sum(1)+1)
    perplexity = - logits.gather(2, preds.unsqueeze(2)).squeeze(2).sum(1) / ((preds>0).to(logits).sum(1)+1)

    return loss, preds, entropy, perplexity




