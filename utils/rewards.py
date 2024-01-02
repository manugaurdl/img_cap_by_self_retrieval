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

    # precomputed document frequencies.
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

    #greedy_res : B cap
    #gen_res : sample_n * B cap
    cider_reward_weight = config['cider_reward_weight']
    bleu_reward_weight = config['bleu_reward_weight']
    batch_size = len(data_gts) 
    gen_result_size = gen_result.shape[0] # no. of policy captions = bsz * 5 cap
    seq_per_img = gen_result_size // len(data_gts) #sample_n
    assert greedy_res.shape[0] == batch_size

    res = OrderedDict()
    gen_result = gen_result.data.cpu().numpy()
    greedy_res = greedy_res.data.cpu().numpy()

    # res : dict of len = B + sample_n * B --> str(tokenized_cap)

    #iterate over B *sample_n policy cap 
    for i in range(gen_result_size):
        res[i] = [array_to_str(gen_result[i])]
    
    #iterate over B greedy cap        
    for i in range(batch_size):
        res[gen_result_size + i] = [array_to_str(greedy_res[i])]
    
    gts = OrderedDict()

    #data_gts : tuple ; len = bsz ; has 5 GTs for each image.
    
    # iterate bsz
    for i in range(len(data_gts)):
        # for each img, iterate 5 gts.
        gts[i] = [array_to_str(data_gts[i][j]) for j in range(len(data_gts[i]))]

    #res_ : list of dicts | number of dicts = (total policy + greedy caps) i.e (sample_n * B + B)
    #res_ created from res.
    #So, First B*sample_n image_ids--> policy captions
    # Last B image_ids -->  greedy captions
    # For bsz = 3 and sample_n = 5 : image_id = 6 is second policy cap of 2nd image and image_id = 17 is greedy cap of 2nd image.

    # sample_n policy captions followed by 1 greedy caption for each image.
    res_ = [{'image_id':i, 'caption': res[i]} for i in range(len(res))]
    res__ = {i: res[i] for i in range(len(res_))}  # same as res_, but a dict (img_id : caption)

    """
    In gts_, each index correspond to [5 gts] for a given image i
    For bsz = 3, sample_n = 5 : len = 18
    gts_ = [i_1,i_1,i_1,i_1,i_1, i_2,i_2,i_2,i_2,i_2, i_3,i_3,i_3,i_3,i_3,  i_1,  i_2,  i_3]
    same structure for res. But first sample_n * B --> policy cap. Last B --> greedy cap.
    """
    gts_ = {i: gts[i // seq_per_img] for i in range(gen_result_size)} # index 1 to 5 --> 1st image's [5 gts], index 5 to 10 : 2nd image's [5 gts]
    gts_.update({i+gen_result_size: gts[i] for i in range(batch_size)}) # index 16 : 1st image's [5 gts].... index 17 :  2rd image's [5 gts]
    
    # reward weight is 1 

    """
    For index i :
        pred = A tokenized caption : str
        target = 5 gt caption for that image.
    
    
    currently,  gts_ : dict of len B * sample_n + B 
                gts_[0] : list of 5 gt for given image.
                gts_[0][0] : 

                res_ : ordered dict of len =  B * sample_n + B 
                res_[0] : generated cap for that image.
    
    """
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
    # import ipdb;ipdb.set_trace()
    scores = cider_reward_weight * cider_scores + bleu_reward_weight * bleu_scores
    """
    For each img :  Take reward for policy cap. Rescale it using reward for greedy cap.
    scores before: (12,), bsz =2

    for each image : CIDEr(policy) - CIDEr(greedy).
    scores (2,5) =      (2,5) -           (2,1)
    greedy cap broadcasted and subtracted from each policy cap
    """
    scores = scores[:gen_result_size].reshape(batch_size, seq_per_img) - scores[-batch_size:][:, np.newaxis]

    # flatten the scaled reward for all policy cap.   
    scores = scores.reshape(gen_result_size)
    
    """
    scores[i] --> reward for policy cap for image i//sample_n.
    logprob of each generated word given same reward. 
    (B* sample_n, 1) rewards --> (B * sample_n, 40)
    """
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