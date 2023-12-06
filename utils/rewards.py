import numpy as np
import time
from collections import OrderedDict
import torch
import sys

try:
    sys.path.append("cider")
    from pyciderevalcap.ciderD.ciderD import CiderD
    from pyciderevalcap.cider.cider import Cider
    sys.path.append("coco-caption")
    from pycocoevalcap.bleu.bleu import Bleu
except:
    print('cider or coco-caption missing')

CiderD_scorer = None
Cider_scorer = None
Bleu_scorer = None
#CiderD_scorer = CiderD(df='corpus')

def init_scorer(cached_tokens):
    global CiderD_scorer
    CiderD_scorer = CiderD_scorer or CiderD(df=cached_tokens)
    global Cider_scorer
    Cider_scorer = Cider_scorer or Cider(df=cached_tokens)
    global Bleu_scorer
    Bleu_scorer = Bleu_scorer or Bleu(4)

def array_to_str(arr):
    out = ''

    for i in range(len(arr)):
        out += str(arr[i]) + ' '
        if arr[i] == 0:
            break
    return out.strip()

def get_self_critical_reward(greedy_res, data_gts, gen_result,config):
    #greedy_res : B,20
    #gen_res : n_sample* B, 20
    cider_reward_weight = config['cider_reward_weight']
    bleu_reward_weight = config['bleu_reward_weight']
    batch_size = len(data_gts) 
    gen_result_size = gen_result.shape[0] # bsz * 5 captions per image
    seq_per_img = gen_result_size // len(data_gts) # gen_result_size  = batch_size * seq_per_img
    assert greedy_res.shape[0] == batch_size

    res = OrderedDict()
    gen_result = gen_result.data.cpu().numpy()
    greedy_res = greedy_res.data.cpu().numpy()

    # res : dict of len = B + n*B
    # it has tokenized caption in str

    for i in range(gen_result_size):
        res[i] = [array_to_str(gen_result[i])]
    for i in range(batch_size):
        res[gen_result_size + i] = [array_to_str(greedy_res[i])]
    
    gts = OrderedDict()

    for i in range(len(data_gts)):
        gts[i] = [array_to_str(data_gts[i][j]) for j in range(len(data_gts[i]))]

    # For batch_size = 1, res_ is a list of 6 image_ids : caption. (5 for REINFORCE, 1 for MLE)
    res_ = [{'image_id':i, 'caption': res[i]} for i in range(len(res))] # list of dicts
    res__ = {i: res[i] for i in range(len(res_))}  # same as res, but a dict (img_id : caption)

    # For bsz = 3 ; total captions = 18 --> 5 policy + 1 greedy  for each image

    gts_ = {i: gts[i // seq_per_img] for i in range(gen_result_size)} # index 1 to 5 --> 1st image's [5 captions], index 5 to 10 : 2nd image's [5 captions]
    gts_.update({i+gen_result_size: gts[i] for i in range(batch_size)}) # index 16 : 1st image's [5 captions].... index 18 :  3rd image's [5 captions]
    
    # reward weight is 1 

    if cider_reward_weight > 0:
        _, cider_scores = CiderD_scorer.compute_score(gts_, res_)
    else:
        cider_scores = 0
    # default = 0
    if bleu_reward_weight > 0:
        _, bleu_scores = Bleu_scorer.compute_score(gts_, res__)
        bleu_scores = np.array(bleu_scores[3])
    else:
        bleu_scores = 0
    scores = cider_reward_weight * cider_scores + bleu_reward_weight * bleu_scores
    # select CIDER scores for policy generated caption and reshape it to (B,seq_per_img) - select CIDER reward for greedy method i.e last B scores.
    # scores before: (12,), bsz =2
    # --> scores (2,5) =  (2,5) - (2,1)

    # for each image : CIDEr(policy) - CIDEr(greedy) for each policy generated caption
    scores = scores[:gen_result_size].reshape(batch_size, seq_per_img) - scores[-batch_size:][:, np.newaxis]
    scores = scores.reshape(gen_result_size)
    #scores = (10,)
    rewards = np.repeat(scores[:, np.newaxis], gen_result.shape[1], 1)

    return rewards

def get_scores(data_gts, gen_result, opt):
    batch_size = gen_result.size(0)# batch_size = sample_size * seq_per_img
    seq_per_img = batch_size // len(data_gts)

    res = OrderedDict()
    
    gen_result = gen_result.data.cpu().numpy()
    for i in range(batch_size):
        res[i] = [array_to_str(gen_result[i])]

    gts = OrderedDict()
    for i in range(len(data_gts)):
        gts[i] = [array_to_str(data_gts[i][j]) for j in range(len(data_gts[i]))]

    res_ = [{'image_id':i, 'caption': res[i]} for i in range(batch_size)]
    res__ = {i: res[i] for i in range(batch_size)}
    gts = {i: gts[i // seq_per_img] for i in range(batch_size)}
    if opt.cider_reward_weight > 0:
        _, cider_scores = CiderD_scorer.compute_score(gts, res_)
    else:
        cider_scores = 0
    if opt.bleu_reward_weight > 0:
        _, bleu_scores = Bleu_scorer.compute_score(gts, res__)
        bleu_scores = np.array(bleu_scores[3])
    else:
        bleu_scores = 0

    scores = opt.cider_reward_weight * cider_scores + opt.bleu_reward_weight * bleu_scores

    return scores

def get_self_cider_scores(data_gts, gen_result, opt):
    batch_size = gen_result.size(0)# batch_size = sample_size * seq_per_img
    seq_per_img = batch_size // len(data_gts)

    res = []
    
    gen_result = gen_result.data.cpu().numpy()
    for i in range(batch_size):
        res.append(array_to_str(gen_result[i]))

    scores = []
    for i in range(len(data_gts)):
        tmp = Cider_scorer.my_self_cider([res[i*seq_per_img:(i+1)*seq_per_img]])
        def get_div(eigvals):
            eigvals = np.clip(eigvals, 0, None)
            return -np.log(np.sqrt(eigvals[-1]) / (np.sqrt(eigvals).sum())) / np.log(len(eigvals))
        scores.append(get_div(np.linalg.eigvalsh(tmp[0]/10)))

    scores = np.array(scores)

    return scores