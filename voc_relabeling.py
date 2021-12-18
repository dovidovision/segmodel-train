import argparse
import numpy as np
from PIL import Image
import os
from glob import glob
import cv2
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Relabeling VOC dataset")
    parser.add_argument('--voc-root',
                        type=str,
                        default = './data/VOCdevkit/VOC2012/SegmentationClass',
                        help="VOC image root, use VOCdevkit/VOC2012/SegmentationClass directory.")
    parser.add_argument('--new-root',
                        type=str,
                        default='./data/VOCdevkit/VOC2012/CatSegmentationClass',
                        help="new root that relabeled segmentaion map is saved.")

    args = parser.parse_args()
    return args

def main(args):
    root = args.voc_root
    new_root = args.new_root
    os.makedirs(new_root,exist_ok=True)
    file_paths = glob(os.path.join(root,'*.png'))

    print('='*100)
    print('Relabeling Start..')
    print(f'root:[{root}]\nnew_root:[{new_root}]\n# images:[{len(file_paths)}]')
    print()
    no_in_cat,in_cat=0,0

    for fpath in tqdm(file_paths):
        image = np.array(Image.open(fpath))
        new_image = (image==8).astype(np.int32)
        if new_image.sum()==0:
            no_in_cat+=1
        else:
            in_cat+=1
            
        new_fname = fpath.split('/')[-1]
        new_fpath = os.path.join(new_root,new_fname)
        cv2.imwrite(new_fpath,new_image)
    print()
    print('Finish Relabeling!!')
    print(f"# image including cat object:[{in_cat}]")
    print(f"# image not including cat object:[{no_in_cat}]")
    print('='*100)

if __name__=='__main__':
    args = parse_args()
    main(args)
