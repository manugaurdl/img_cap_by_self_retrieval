import torch
import skimage.io as io
import clip
from PIL import Image
import pickle
import json
import os
from tqdm import tqdm
import argparse


def main(clip_model_type: str, split : str):
    device = torch.device('cuda:0')
    clip_model_name = clip_model_type.replace('/', '_')
    out_path = f"/ssd_scratch/cvit/manu/clipcap/parse_coco/{clip_model_name}_{split}_emb.pkl"
    clip_model, preprocess = clip.load(clip_model_type, device=device, jit=False)
    
    with open(f'/ssd_scratch/cvit/manu/img_cap_self_retrieval_clip/data/parse_coco_req/{split}_cocoids.json', 'r') as f:
        data = json.load(f)
    
    print("%0d captions loaded from json " % len(data))
    all_embeddings = []
    all_captions = []
    for i in tqdm(range(len(data))):
        
        d ={}
        cocoid = data[i]
        d['cocoid'] = cocoid
        
        filename = f"/ssd_scratch/cvit/manu/coco/train2014/COCO_train2014_{int(cocoid):012d}.jpg"
        if not os.path.isfile(filename):
            filename = f"/ssd_scratch/cvit/manu/coco/val2014/COCO_val2014_{int(cocoid):012d}.jpg"

        image = io.imread(filename)
        image = preprocess(Image.fromarray(image)).unsqueeze(0).to(device)
        with torch.no_grad():
            prefix = clip_model.encode_image(image).cpu()

        d["clip_embedding"] = i
        all_embeddings.append(prefix)
        all_captions.append(d)

        if (i + 1) % 10000 == 0:
            with open(out_path, 'wb') as f:
                pickle.dump({"clip_embedding": torch.cat(all_embeddings, dim=0), "images": all_captions}, f)

    with open(out_path, 'wb') as f:
        pickle.dump({"clip_embedding": torch.cat(all_embeddings, dim=0), "images": all_captions}, f)

    print('Done')
    print("%0d embeddings saved " % len(all_embeddings))
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--clip_model_type', default="ViT-B/32", choices=('RN50', 'RN101', 'RN50x4', 'ViT-B/32'))
    parser.add_argument('--split' , type = str, choices = ("train", "val", "test"))
    args = parser.parse_args()
    exit(main(args.clip_model_type, args.split))