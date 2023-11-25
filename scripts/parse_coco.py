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

SAVE = False

class CocoDataset(Dataset):
    """
    input : {split}_caption.json

    """
    def __init__(self, data_dir,split,model,transforms):
        self.data_dir = data_dir
        self.split = split
        self.model = model
        self.transforms = transforms
        with open(f'/ssd_scratch/cvit/manu/img_cap_self_retrieval_clip/annotations/{self.split}_caption.json', 'r') as f:
            self.data_list = json.load(f)
        print(f"%0d captions loaded from {self.split} json " % len(self.data_list))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self,idx):

        data = self.data_list[idx]
        img_id = data['image_id']
        caption = data['caption']

        #get img_path
        filename = os.path.join(self.data_dir,f"train2014/COCO_train2014_{int(img_id):012d}.jpg")
        if not os.path.isfile(filename):
            filename = os.path.join(self.data_dir,f"val2014/COCO_val2014_{int(img_id):012d}.jpg")
        try:
            assert os.path.isfile(filename)==True
        except:
            print(f"image for {filename} doesn't exist")
        if self.model == "clip":
            image = io.imread(filename)
            image = self.transforms(Image.fromarray(image))
            
        elif self.model =="siglip":
            image = Image.open(filename)
            image = self.transforms(image.convert('RGB'))

        # return {'caption' : caption, 'image' : image}
        return (caption, image)

def main(args):
    data_dir = args.data_dir
    split = args.split
    batch_size = args.bs
    print(f'split = {args.split}')
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    out_path = os.path.join(data_dir,f"{args.model}_{args.clip_model_type.replace('/','_')}_{split}_emb.pkl")

    all_embeddings = [] # all images clip embeddings stored 
    all_captions = []  # stores data dict for all images. also stores idx for clip_embedding
    
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
    if args.model=="clip":
        dataset = CocoDataset(data_dir, split, args.model, preprocess)
    else:
        dataset = CocoDataset(data_dir, split, args.model,transforms)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    for idx, (captions, images) in tqdm(enumerate(dataloader), total=len(dataloader)):

        with torch.no_grad():
            if args.model == "clip":
                import pdb;pdb.set_trace()
                prefix = clip_model.encode_image(images.to(device)).cpu()
                
            elif args.model =="siglip":    
                prefix = model(images)  # output is (batch_size, num_features) shaped tensor

        all_embeddings.extend(torch.split(prefix, 1, dim=0))
        all_captions.extend(captions)
        if SAVE:
            if ((idx+1)*args.bs) % int(args.bs)*100== 0:
                with open(out_path, 'wb') as f:
                    pickle.dump({"clip_embedding": torch.cat(all_embeddings, dim=0), "captions": all_captions}, f)
        gc.collect()

    if SAVE:
        with open(out_path, 'wb') as f:
            pickle.dump({"clip_embedding": torch.cat(all_embeddings, dim=0), "captions": all_captions}, f)

    print('Done')
    print("%0d embeddings saved " % len(all_embeddings))
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default="clip", choices=("siglip", "clip"))
    parser.add_argument('--clip_model_type', default="ViT-B/32", choices=('RN50', 'RN101', 'RN50x4', 'ViT-B/32', 'ViT-B/16'))
    parser.add_argument('--data_dir', default="/ssd_scratch/cvit/manu/coco/")
    parser.add_argument('--split', default = 'val',type = str)
    parser.add_argument('--bs', default = 64, type = int)

    
    args = parser.parse_args()
    exit(main(args))
