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
from tqdm import tqdm 
from utils.helper_functions import *

# load coco-caption if available
try:
    sys.path.append("coco-caption")
    from pycocotools.coco import COCO
    from pycocoevalcap.eval import COCOEvalCap
except Exception as e:
    print('Warning: coco-caption not available. Error meesage:', e)


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


@torch.no_grad()
def validation(model, val_dataloader,val_dataset, device, eval_obj,config):

    tokenizer = val_dataset.tokenizer
    max_length = val_dataset.max_len_token
    top_p= config['top_p']
    temp = config['temp']
    lang_eval = config['lang_eval']
    loss_meter = AverageMeter("eval_loss", ":.5f")
    method = config['method']
    sampling_method = config['sampling_method']
    stop_token =  tokenizer.encode(config['stop_token'])[0]

    generated_num = 0
    generated_list = []
    
    #additional args
    dataset ='cocotalk.json'


    val_preds = []
    val_targets = []
    
    model.eval()
    predictions = []
    # for idx, (prefix, targets, mask) in tqdm(enumerate(val_dataloader), total=len(val_dataloader)):


    for idx, (prefix, targets, mask, meta_data) in enumerate(val_dataloader):
        print(f'sampled batch no.{idx} in val dataloader')
        # meta_data --> cocoid, filename, sentence_id
        targets, mask, prefix = targets.to(device), mask.to(device), prefix.to(device, dtype=torch.float32)
        #get the class idx for each instance in the batch.
        prefix_embed = model(prefix,targets.to(device), mask.to(device), only_prefix = True)

        preds, seqLogprob, tokens  = sample(max_length, prefix_embed, model, temp, sampling_method, stop_token)  
        tokens = tokens.data
        #loss meter updated for each batch
        logits  = torch.cat((preds), dim = 1) # (B,max_caption_len, vocab_size)
        

        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.to(torch.long).flatten(), ignore_index = 0)
        loss_meter.update(loss.item(), targets.shape[0])

        entropy = -(F.softmax(logits, dim=2) * logits).sum(2).sum(1) / ((tokens>0).to(logits).sum(1)+1)
        perplexity = - logits.gather(2, tokens.unsqueeze(2)).squeeze(2).sum(1) / ((tokens>0).to(logits).sum(1)+1)

        # generated captions of each batch added to list --> gen captions for whole split
        val_preds.append(tokens)
        val_targets.append(targets)
    
    #--------------------------------------------------------
        sents = tokenizer.batch_decode(tokens)
        # predictions --> [{'image_id': 184613, 'caption': 'a swimmer ravine fee...iers backs', 'perplexity': 8.26982307434082, 'entropy': 8.715027809143066}]
        for k, sent in enumerate(sents): # sents is a list of batch_size length.

            # entry = {'image_id': data['infos'][k]['id'], 'caption': sent, 'perplexity': perplexity[k].item(), 'entropy': entropy[k].item()}
            entry = {'image_id' : meta_data['cocoid'][k].item(), 'caption': sent, 'perplexity': perplexity[k].item(), 'entropy': entropy[k].item()}

            # if eval_kwargs.get('dump_path', 0) == 1:
            #     entry['file_name'] = data['infos'][k]['file_path'] # k --> index of batch to get same index infos
            predictions.append(entry)
            # if eval_kwargs.get('dump_images', 0) == 1:
            #     # dump the raw image to vis/ folder
            #     cmd = 'cp "' + os.path.join(eval_kwargs['image_root'], data['infos'][k]['file_path']) + '" vis/imgs/img' + str(len(predictions)) + '.jpg' # bit gross
            #     print(cmd)
            #     os.system(cmd)

            # if verbose:
            #     print('image %s: %s' %(entry['image_id'], entry['caption']))

        # if sample_n > 1:
        #     eval_split_n(model, n_predictions, [fc_feats, att_feats, att_masks, data], eval_kwargs)
        
        # ix0 = data['bounds']['it_pos_now']
        # ix1 = data['bounds']['it_max']
        # if num_images != -1:
        #     ix1 = min(ix1, num_images)
        # else:
        #     num_images = ix1
        # for i in range(n - ix1):
        #     predictions.pop()

        # if verbose:
        #     print('evaluating validation preformance... %d/%d (%f)' %(n, ix1, loss))
        break
    lang_stats = None

    # #What are we saving

    # if not os.path.isdir('eval_results'):
    #     os.mkdir('eval_results')
    # torch.save((predictions, n_predictions), os.path.join('eval_results/', '.saved_pred_'+ eval_kwargs['id'] + '_' + split + '.pth'))
    
    if lang_eval == 1:
        lang_stats = language_eval(dataset, predictions, method)

    # # Switch back to training mode
    # model.train()
    # return loss_sum/loss_evals, predictions, lang_stats
#--------------------------------------------------------


    # eval metrics for val set 
    # pred_cap = tokenizer.batch_decode(torch.cat((val_preds), dim=0))
    # val_targets = torch.cat((val_targets), dim=0)
    # mask = val_targets>0
    # target_cap  = [[tokenizer.decode(val_targets[i][mask[i]])] for i in range(val_targets.shape[0])]
    # bleu_score = eval_obj.get_metric('bleu',pred_cap, target_cap )['bleu']
    # meteor_score  = eval_obj.get_metric('meteor',pred_cap, target_cap)['meteor']
        
    return loss_meter, lang_stats

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


def language_eval(dataset, preds, method, split='val'):
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
    valids = coco.getImgIds() #list of 40504 ids (test + val + restval)

    # consider preds only if it exists in MSCOCO validation set
    preds_filt = [p for p in preds if p['image_id'] in valids]
    mean_perplexity = sum([_['perplexity'] for _ in preds]) / len(preds)
    mean_entropy = sum([_['entropy'] for _ in preds]) / len(preds)
    print('using %d/%d predictions' % (len(preds_filt), len(preds)))
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

    for p in preds_filt:
        image_id, caption = p['image_id'], p['caption']
        imgToEval[image_id]['caption'] = caption

    out['bad_count_rate'] = sum([count_bad(_['caption']) for _ in preds_filt]) / float(len(preds_filt))
    outfile_path = os.path.join('eval_results/', method + '_' + split + '.json')
    with open(outfile_path, 'w') as outfile:
        json.dump({'overall': out, 'imgToEval': imgToEval}, outfile)

    return out


#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def sample(max_length, token_emb, model,temp, method, stop_token, tokens = None):
    """
    seq : sampled sequence (B, max_len)
    seqLogprobs : logprobs for sampled words
    token_emb : initially the image embedding
    method : greedy, sample

    no need for return token embedding
    """
    preds = []
    unfinished = torch.ones((token_emb.shape[0],1), dtype=torch.bool) # all are unfinished i.e True at start
    # each iteration, the context on which generation is conditioned increases by 1. (prefix + gpt_outputs)
    for t in range(round(float(max_length))):

        outputs = model.gpt(inputs_embeds= token_emb)
        #LM head output --> distribution over vocab
        logits = outputs.logits # (B, prefix_len, vocab_size)    
        # logit of last token = next token prediction
        logits =  logits[:, -1, :]/ (temp if temp > 0 else 1.0)  # (B,vocab_size)
        # preds for timestep t
        preds.append(logits.unsqueeze(1)) #(B,1,vocab_size)
        if method == "greedy":
            sampled_logprob, next_token = torch.max(logits.data,dim = -1)
        elif method == "sample":
            probs = torch.nn.functional.softmax(logits, dim=-1) # (B, C)
            next_token = torch.multinomial(probs, num_samples=1) # (B, 1)
            sampled_logprob = logits.gather(1, next_token)

        # for the sampled token, get token embedding 
        next_token_embed = model.gpt.transformer.wte(next_token) # (B,1,768)

        if tokens is None:
            tokens = next_token
            seqLogprob = sampled_logprob
        else:
            tokens = torch.cat((tokens, next_token), dim=1)
            seqLogprob = torch.cat((seqLogprob, sampled_logprob), dim = 1)
            print(tokens.shape)
            print(seqLogprob.shape)
        token_emb = torch.cat((token_emb, next_token_embed), dim=1)

        # if stop token reached for all images --> break        
        unfinished[torch.nonzero(next_token==stop_token)] = False

        if sum(unfinished).item() == 0:
            break
    return preds, seqLogprob, tokens 

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
