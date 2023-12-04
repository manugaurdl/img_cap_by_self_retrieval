import torch
import skimage.io as io
import clip
from PIL import Image
import pickle
import json
import os
from tqdm import tqdm
import argparse
import timm
import gc
from torch.utils.data import Dataset, DataLoader
import pdb
import time
import logging

SAVE = True
# Setting up logging

formatter = logging.Formatter('%(message)s')


def setup_logger(name, log_file, level=logging.DEBUG):
    """ setup as many loggers as you want"""

    handler = logging.FileHandler(log_file)        
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger

# last idx saved logger
log_last_idx = setup_logger('log_last_idx', '../logs/parse_coco/last_idx_saved.log')

# corrupt/missing images logger
image_issue_logger = setup_logger('image_issue_logger', '../logs/parse_coco/image_issue_log.log')

def another_method():
   # using logger defined above also works here
   logger.info('Inside method')

class CocoDataset(Dataset):
    """
    input : {split}_caption.json

    """
    def __init__(self, data_dir,split,model,transforms):
        self.data_dir = data_dir
        self.split = split
        self.model = model
        self.transforms = transforms
        with open(f'/ssd_scratch/cvit/manu/img_cap_self_retrieval_clip/data/parse_coco_req/{self.split}_coco_dataset.json', 'r') as f:
            self.data_list = json.load(f)
        print(f"%0d captions loaded from {self.split} json " % len(self.data_list))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self,idx):

        data = self.data_list[idx]
        cocoid = data['cocoid']
        caption = data['caption']
        sent_id = data['sent_id']
        filename = data['filename']

        #get img_path
        filename = os.path.join(self.data_dir,f"train2014/COCO_train2014_{int(cocoid):012d}.jpg")
        if not os.path.isfile(filename):
            filename = os.path.join(self.data_dir,f"val2014/COCO_val2014_{int(cocoid):012d}.jpg")
        try:
            assert os.path.isfile(filename)==True
        except:
            image_issue_logger.debug(f"image for {filename} doesn't exist")
            
        if self.model == "clip":
            image = io.imread(filename)
            image = self.transforms(Image.fromarray(image))
            
        elif self.model =="siglip":
            image = Image.open(filename)
            image = self.transforms(image.convert('RGB'))

        # return {'caption' : caption, 'image' : image}
        return {'cocoid' : cocoid,  
                "caption" : caption,
                "sent_id" : sent_id,
                "filename" : filename,
                "image" : image
                }

def main(args):
    data_dir = args.data_dir
    split = args.split
    batch_size = args.bs
    print(f'split = {args.split}')
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    out_path = os.path.join(data_dir,f"{args.model}_{args.clip_model_type.replace('/','_')}_{split}_emb.pkl")  # {model}_{split}_emb.pkl

    all_embeddings = [] # all images clip embeddings stored 
    all_captions = []  # stores data dict for all images. also stores idx for clip_embedding
    all_cocoids = []
    all_filenames = []
    all_sentids = []
    #Model Instantiation
    if args.model == "clip":
        clip_model_type = args.clip_model_type
        clip_model_name = clip_model_type.replace('/', '_')
        clip_model, preprocess = clip.load(clip_model_type, device=device, jit=False)

    elif args.model == 'siglip':
        model = timm.create_model(
            'vit_base_patch16_siglip_224',
            pretrained=True,
            num_classes=0,
        )
        model = model.eval()
        data_config = timm.data.resolve_model_data_config(model)
        transforms = timm.data.create_transform(**data_config, is_training=False)
    # Dataset init
    if args.model=="clip":
        dataset = CocoDataset(data_dir, split, args.model, preprocess)
    else:
        dataset = CocoDataset(data_dir, split, args.model,transforms)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    for idx, data_dict in tqdm(enumerate(dataloader), total=len(dataloader)):
        # if script crashes. Can resume s.t idx>logged_idx

        cocoid = data_dict['cocoid'].tolist()
        caption = data_dict['caption']
        sent_id = data_dict['sent_id'].tolist()
        filename = data_dict['filename']
        image = data_dict['image']

        with torch.no_grad():
            if args.model == "clip":
                prefix = clip_model.encode_image(image.to(device)).cpu()
                
            elif args.model =="siglip":    
                prefix = model(image)  # output is (batch_size, num_features) shaped tensor

        all_embeddings.extend(torch.split(prefix, 1, dim=0))
        all_captions.extend(caption)
        all_cocoids.extend(cocoid)
        all_filenames.extend(filename)
        all_sentids.extend(sent_id)

        if SAVE:
            if idx%100== 0:
                with open(out_path, 'wb') as f:
                    pickle.dump({"clip_embedding": torch.cat(all_embeddings, dim=0), 
                                "caption": all_captions,
                                "cocoid" : all_cocoids,
                                "filename": all_filenames,
                                "sent_id" : all_sentids
                                }, f)
                #log last_idx savedo
                log_last_idx.debug(f'{split}.{idx}')
        gc.collect()

    if SAVE:
        # for large train files 2 .pkl saved : a) incremental b) final 
        if split == 'train':
            out_path = os.path.join(data_dir,f"final_{args.model}_{args.clip_model_type.replace('/','_')}_{split}_emb.pkl") 
        with open(out_path, 'wb') as f:
            pickle.dump({"clip_embedding": torch.cat(all_embeddings, dim=0), 
            "caption": all_captions,
            "cocoid" : all_cocoids,
            "filename": all_filenames,
            "sent_id" : all_sentids
            }, f)
    print('Done')
    print("%0d embeddings saved " % len(all_embeddings))
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default="clip", choices=("siglip", "clip"))
    parser.add_argument('--clip_model_type', default="ViT-B/32", choices=('RN50', 'RN101', 'RN50x4', 'ViT-B/32', 'ViT-B/16'))
    parser.add_argument('--data_dir', default="/ssd_scratch/cvit/manu/coco/")
    parser.add_argument('--split', default = 'train',type = str)
    parser.add_argument('--bs', default = 64, type = int)

    
    args = parser.parse_args()
    exit(main(args))
