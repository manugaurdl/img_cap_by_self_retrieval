import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from json import encoder
import random
import string
import time
import os
import sys
import pickle
from tqdm import tqdm 
from utils.helper_functions import *
# load coco-caption if available

# sys.path.append("coco-caption")
# from pycocotools.coco import COCO
# from pycocoevalcap.eval import COCOEvalCap

from argparse import ArgumentParser

from pprint import pprint
from pycocoevalcap.eval import Bleu, Cider, Meteor, PTBTokenizer, Rouge, Spice

bad_endings = ['a','an','the','in','for','at','of','with','before','after','on','upon','near','to','is','are','am']
bad_endings += ['the']


def count_bad(sen):
    sen = sen.split(' ')
    if sen[-1] in bad_endings:
        return 1
    else:
        return 0
"""
eval config
verbose : True 
"""

def getCOCO(dataset):
    if 'coco' in dataset:
        annFile = 'coco-caption/annotations/captions_val2014.json'
    elif 'flickr30k' in dataset or 'f30k' in dataset:
        annFile = 'data/f30k_captions4eval.json'
    return COCO(annFile)

def pad(tokens, padding):
    pad = torch.zeros(padding)
    pad = pad.masked_fill(pad ==0, -1)
    tokens = torch.cat((tokens, pad)).int()
    return tokens

@torch.no_grad()
def validation(model, val_dataloader,val_dataset, device, config):
    
    tokenizer = val_dataset.tokenizer
    max_length = val_dataset.max_len_token
    top_p= config['top_p']
    temp = config['temp']
    lang_eval = config['lang_eval']
    loss_meter = AverageMeter("eval_loss", ":.5f")
    method = config['method']
    sampling_method = config['sampling_method']
    stop_token =  tokenizer.encode(config['stop_token'])[0]
    eval_sample_n = config['eval_sample_n']
    generated_num = 0
    generated_list = []
    repeat_num = 0
    #additional args
    dataset ='cocotalk.json'


    # val_preds = []
    # val_targets = []
    
    model.eval()
    # predictions = []
    # for idx, (prefix, targets, mask) in tqdm(enumerate(val_dataloader), total=len(val_dataloader)):
    
    step_time_avg = []
    cocoid2pred = {}

    for idx, (prefix, targets, mask, untokenized_cap, meta_data) in tqdm(enumerate(val_dataloader), total=len(val_dataloader)):
        # step_time_start = time.time()

        if idx ==0 and eval_sample_n > 1:
            repeat_num = logits.shape[0]//targets.shape[0] 

        targets, mask, prefix = targets.to(device), mask.to(device), prefix.to(device, dtype=torch.float32)

        if config['reproduce_clipcap']:
            prefix_embed = model.clip_project(prefix).view(prefix.shape[0],model.prefix_length, -1)
        else:
            prefix_embed = model(prefix,targets, mask, only_prefix = True)

        #sample entire caption
        # preds : last token's logit at every time step
        preds, seqLogprob, tokens  = sample(max_length, prefix_embed, model, temp, sampling_method, stop_token,tokenizer, config, sample_n = eval_sample_n)
        tokens = tokens.data
    
        logits  = torch.cat((preds), dim = 1) # (B, K , vocab_size) ; K --> max_cap_length for the batch of sampled captions

        # if eval_sample_n > 1: 
        #     #currently not using filename, sent_id hence not repeated
        #     meta_data['cocoid'] = repeat_tensors(repeat_num, meta_data['cocoid'])
        #     targets = repeat_tensors(repeat_num, targets)

        ## making targets and logits of equal len ------------------
        
        # if targets.shape[-1] != logits.shape[1]:
        #     # pad logits --> max_len
        #     padding = targets.shape[1] - logits.shape[1]
        #     padding_config = (0, 0, 0, padding)  # (left, right, top, bottom)

        #     logits = F.pad(logits, padding_config)
            

        # loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.to(torch.long).flatten(), ignore_index = 0)
        # loss_meter.update(loss.item(), targets.shape[0])

        entropy = -(F.softmax(logits, dim=2) * logits).sum(2).sum(1) / ((tokens>0).to(logits).sum(1)+1)
        perplexity = - logits.gather(2, tokens.unsqueeze(2).to(torch.int64)).squeeze(2).sum(1) / ((tokens>0).to(logits).sum(1)+1)

        # generated captions of each batch added to list --> gen captions for whole split
        # val_preds.append(tokens)
        # val_targets.append(targets)
    
    #--------------------------------------------------------
        sents = tokenizer.batch_decode(tokens)
        sents = [[sent.split("!")[0]] for sent in sents]
        data = {}
        
        for i, caption in enumerate(sents):
            data[meta_data[i].item()] = [caption]
        
        cocoid2pred.update(data)    
        if idx < 3: 
            print(cocoid2pred)

        # predictions --> [{'image_id': 184613, 'caption': 'a swimmer ravine fee...iers backs', 'perplexity': 8.26982307434082, 'entropy': 8.715027809143066}]

        # for k, sent in enumerate(sents): # sents is a list of batch_size length.
        #     # entry = {'image_id': data['infos'][k]['id'], 'caption': sent, 'perplexity': perplexity[k].item(), 'entropy': entropy[k].item()}
        #     entry = {'image_id' : meta_data[k].item(), 'caption': sent, 'perplexity': perplexity[k].item(), 'entropy': entropy[k].item()}

        #     predictions.append(entry)

        # if sample_n > 1:
        #     eval_split_n(model, n_predictions, [fc_feats, att_feats, att_masks, data], eval_kwargs)

    # with open("/ssd_scratch/cvit/manu/val_preds_temp.pkl", "wb") as f:
    #     pickle.dump(cocoid2pred, f)

    idx2cocoid = {idx : cocoid for idx, cocoid in enumerate(cocoid2pred.keys())}
    
    path = os.path.join(config['data_dir'], "data/cocoid2caption.pkl")

    with open(path, "rb") as f:
        all_gts = pickle.load(f)
    
    val_cocoids = cocoid2pred.keys()
    
    val_gt = {}
    for cocoid, captions in all_gts.items():
        if cocoid in val_cocoids:
            val_gt[cocoid] = captions
    
    gold_standard = {}

    for i in range(len(cocoid2pred)):
        gold_standard[i] = [{"caption" : c} for c in val_gt[idx2cocoid[i]]]
    
    predictions = {}
    for i in range(len(cocoid2pred)):
        predictions[i] = [{"caption" : cocoid2pred[idx2cocoid[i]][0][0]}]
        
    
    lang_stats = None
    

    # if not os.path.isdir('eval_results'):
    #     os.mkdir('eval_results')
    # torch.save((predictions, n_predictions), os.path.join('eval_results/', '.saved_pred_'+ eval_kwargs['id'] + '_' + split + '.pth'))
    
    if lang_eval == 1:
        # lang_stats = language_eval(dataset, predictions, method, 'val')
        lang_stats = compute_nlg_metrics(predictions,gold_standard)
        import ipdb;ipdb.set_trace()
    return loss_meter, lang_stats

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


def compute_nlg_metrics(predictions, gold_standard):
    tokenizer = PTBTokenizer()
    predictions = tokenizer.tokenize(predictions)
    ground_truth = tokenizer.tokenize(gold_standard)

    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr"),
        (Spice(), "SPICE"),
    ]

    summary = {}
    for scorer, method in scorers:
        score, scores = scorer.compute_score(ground_truth, predictions)
        if isinstance(method, list):
            for sc, scs, m in zip(score, scores, method):
                summary[m] = sc
        else:
            summary[method] = score
    print()
    pprint(summary)
    return summary


def language_eval(dataset, preds, method, split):
    """
    dataset : cocotalk.json
    preds : list of 5k pred dicts {image_id, caption, perplexity, entropy}
    
    OUT --> 'eval_results/', method + '_' + split + '.json' {predictions and eval metric scores}
    """

    
    # model_id = eval_kwargs['id']
    # eval_oracle = eval_kwargs.get('eval_oracle', 0)
    
    # create output dictionary
    out = {}

    if not os.path.exists('eval_results/'):
        os.makedirs('eval_results/')
    cache_path = os.path.join('eval_results/', '.cache_coco' + '_' + split + '.json')

    coco = getCOCO(dataset) #pycocotools
    
    # valids = coco.getImgIds() #list of 40504 ids (test + val + restval)

    # consider preds only if it exists in MSCOCO validation set
    # preds_filt = [p for p in preds if p['image_id'] in valids]
    mean_perplexity = sum([_['perplexity'] for _ in preds]) / len(preds)
    mean_entropy = sum([_['entropy'] for _ in preds]) / len(preds)
    # print('using %d/%d predictions' % (len(preds), len(preds)))
    json.dump(preds, open(cache_path, 'w')) # serialize to temporary json file. Sigh, COCO API...

    cocoRes = coco.loadRes(cache_path) #Load algorithm results and create API for accessing them.
    cocoEval = COCOEvalCap(coco, cocoRes) #
    cocoEval.params['image_id'] = cocoRes.getImgIds()
    cocoEval.evaluate()

    for metric, score in cocoEval.eval.items():
        out[metric] = score
    # Add mean perplexity
    out['perplexity'] = mean_perplexity
    out['entropy'] = mean_entropy

    imgToEval = cocoEval.imgToEval
    for k in list(imgToEval.values())[0]['SPICE'].keys():
        if k != 'All':
            out['SPICE_'+k] = np.array([v['SPICE'][k]['f'] for v in imgToEval.values()])
            out['SPICE_'+k] = (out['SPICE_'+k][out['SPICE_'+k]==out['SPICE_'+k]]).mean()

    for p in preds:
        image_id, caption = p['image_id'], p['caption']
        imgToEval[image_id]['caption'] = caption

    out['bad_count_rate'] = sum([count_bad(_['caption']) for _ in preds]) / float(len(preds))
    outfile_path = os.path.join('eval_results/', method + '_' + split + '.json')
    with open(outfile_path, 'w') as outfile:
        json.dump({'overall': out, 'imgToEval': imgToEval}, outfile)

    return out


#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def sample(max_length, token_emb, model, temp, method, stop_token, tokenizer, config, sample_n = None, tokens = None):
    """
    seq : sampled sequence (B, max_len)
    seqLogprobs : logprobs for sampled words
    token_emb : initially the image embedding
    method : greedy, sample

    returns : preds (last token logit at every time step.)
    """
    max_len = round(float(max_length))
    if method == "greedy":
        sample_n = 1
    
    pred_logits = []        
    # for sample_n > 1 : repeat images --> batch size increase sample_n X times 
    if method =="sample" and sample_n > 1:
        token_emb = repeat_tensors(sample_n, token_emb)
    
    #(B,40)
    tokens = torch.zeros((token_emb.shape[0], max_len), dtype = torch.int).to(token_emb.device)
    seqLogprob = torch.zeros((token_emb.shape[0], max_len)).to(token_emb.device)
    
    # each iteration, the context on which generation is conditioned increases by 1. (prefix + gpt_outputs)
    # T1 = time.time()
    for t in range(max_len):

        if t == max_len:
            break
        outputs = model.gpt(inputs_embeds= token_emb)

        
        #LM head output --> distribution over vocab
        logits = outputs.logits # (B, prefix_len, vocab_size)    
        # logit of last token = next token prediction
        logits =  logits[:, -1, :]/ (temp if temp > 0 else 1.0)  # (B,vocab_size)
        # preds for timestep t
        pred_logits.append(logits.unsqueeze(1)) #(B,1,vocab_size)

        if method == "greedy":
            sampled_logprob, next_token = torch.max(logits.data,dim = -1)        

        elif method == "sample":
            # probs = torch.nn.functional.softmax(logits.data, dim=-1) # (B, vocab_size)
            logprobs = torch.nn.functional.log_softmax(logits.data, dim=-1) # (B, vocab_size)
            # next_token = torch.multinomial(logprobs, num_samples=1).squeeze(-1) # (B, 1)
            next_token = torch.distributions.Categorical(logits=logprobs.detach()).sample()
            sampled_logprob = logits.gather(1, next_token.clone().unsqueeze(-1)).squeeze(-1)

            #************************************************************************

        if t ==0:
            # True for indices where stop token is not reached.
            try:
                unfinished = next_token != stop_token
            except:
                import ipdb;ipdb.set_trace()
        else:
            # For instances in batch which are finished --> overwrite sampled next_token with 0.
            next_token[~unfinished] = 0
            sampled_logprob[~unfinished] = 0

            # logprobs = logprobs * unfinished.unsqueeze(1).to(logprobs)

            # zero out log probs?
            # If stop_token reached for an idx, unfinished = False
            unfinished = unfinished & (next_token != stop_token)        
        
        # t_th index  = token sampled for t_th index
        tokens[:,t] = next_token
        seqLogprob[:,t] = sampled_logprob
        
        # for the sampled token, get token embedding 
        next_token_embed = model.gpt.transformer.wte(next_token.unsqueeze(-1)) # (B,1,768)
        token_emb = torch.cat((token_emb, next_token_embed), dim=1)

        # unfinished[torch.nonzero((next_token==stop_token).squeeze(-1)).cpu()] = False
        
        if unfinished.sum() == 0:
            return pred_logits, seqLogprob[:,:t+1], tokens[:,:t+1]

    # if method=="sample":
    #     step_time_avg.append(time.time() - T1)
    #     print(len(step_time_avg))
    #     print(f"bsz {config['batch_size']} sample_n {config['train_sample_n']} : {np.mean(np.array(step_time_avg))}")
    return pred_logits, seqLogprob, tokens 

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------