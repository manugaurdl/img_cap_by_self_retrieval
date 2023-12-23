import json
import pickle
import torch
import os

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

def save_model(output_dir, model_name, model, optimizer, epoch):
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join(output_dir, f'{model_name}.pt'))


def load_model(model, output_dir, model_name):
    path = os.path.join(output_dir, f'{model_name}.pt')
    model.load_state_dict(torch.load(path)['model_state_dict'])

def int2mil(number):
    if abs(number) >= 1_000_000:
        formatted_number = "{:.1f}M".format(number / 1_000_000)
    else:
        formatted_number = str(number)
    return formatted_number

def trainable_params(model):
    print(f'{int2mil(sum(p.numel() for p in model.parameters() if p.requires_grad == True))} trainable params')

def open_json(path):
    with open(path, 'r') as f:
        file = json.load(f)
    return file


# Input: seq, N*D numpy array, with element 0 .. vocab_size. 0 is END token.
def decode_sequence(ix_to_word, seq):
    N, D = seq.size()
    out = []
    for i in range(N):
        txt = ''
        for j in range(D):
            ix = seq[i,j]
            if ix > 0 :
                if j >= 1:
                    txt = txt + ' '
                try:
                    txt = txt + ix_to_word[str(ix.item())]
                except:
                    import ipdb;ipdb.set_trace()

            else:
                break
        if int(os.getenv('REMOVE_BAD_ENDINGS', '0')):
            flag = 0
            words = txt.split(' ')
            for j in range(len(words)):
                if words[-j-1] not in bad_endings:
                    flag = -j
                    break
            txt = ' '.join(words[0:len(words)+flag])
        out.append(txt.replace('@@ ', ''))
    return out

def repeat_tensors(n, x):
    """
    For a tensor of size Bx..., we repeat it n times, and make it Bnx...
    For collections, do nested repeat
    """
    if torch.is_tensor(x):
        x = x.unsqueeze(1) # Bx1x...
        x = x.expand(-1, n, *([-1]*len(x.shape[2:]))) # Bxnx...
        x = x.reshape(x.shape[0]*n, *x.shape[2:]) # Bnx...
    elif type(x) is list or type(x) is tuple:
        x = [repeat_tensors(n, _) for _ in x]
    return x


def split_tensors(n, x):
    if torch.is_tensor(x):
        assert x.shape[0] % n == 0
        x = x.reshape(x.shape[0] // n, n, *x.shape[1:]).unbind(1)
    elif type(x) is list or type(x) is tuple:
        x = [split_tensors(n, _) for _ in x]
    elif x is None:
        x = [None] * n
    return x


def set_data_dir(config):

    if config['jatayu']:
        data_dir = '/home/manugaur/img_cap_self_retrieval/data'
    else:
            data_dir = '/ssd_scratch/cvit/manu/img_cap_self_retrieval_clip'
    
    config['train_data'] = os.path.join(data_dir, config['train_data'])
    config['val_data'] = os.path.join(data_dir, config['val_data'])
    config['test_data'] = os.path.join(data_dir, config['test_data'])
    config['out_dir'] = os.path.join(data_dir, config['out_dir'])
    config['cocotalk'] = os.path.join(data_dir, config['cocotalk'])
    config['data_dir'] = data_dir