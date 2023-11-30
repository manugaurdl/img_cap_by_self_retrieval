import json
import pickle
import torch
import os
import evaluate

# def save_config(args: argparse.Namespace):
#     config = {}
#     for key, item in args._get_kwargs():
#         config[key] = item
#     out_path = os.path.join(args.out_dir, f"{args.prefix}.json")
#     with open(out_path, 'w') as outfile:
#         json.dump(config, outfile)

class Summary():
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.val_history = list()
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.val_history = list()

    def update(self, val, n=1):
      #n : batch size
      #val :avg loss
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.val_history.append(val) # maintaining a list of val losses.        

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)
        return fmtstr.format(**self.__dict__)


def open_pickle(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

def dump_pickle(data,path):
    with open(path, 'wb') as f:
        pickle.dump(data,f)

def save_model(output_dir, model_name, model):
    torch.save(
    model.state_dict(),
    os.path.join(output_dir, f'{model_name}.pt'),
)

class Metrics(object):
    def __init__(self):
        
        self.bleu =  evaluate.load("bleu")
        self.meteor = evaluate.load("meteor")
    
    def get_metric(self, metric,preds,refs):
        if metric == "bleu":
            return self.bleu.compute(predictions=preds, references= refs)
        elif metric == "meteor":
            return self.meteor.compute(predictions=preds, references= refs)

def int2mil(number):
    if abs(number) >= 1_000_000:
        formatted_number = "{:.1f}M".format(number / 1_000_000)
    else:
        formatted_number = str(number)
    return formatted_number

def trainable_params(model):
    print(f'{int2mil(sum(p.numel() for p in model.parameters() if p.requires_grad == True))} trainable params')
