import numpy as np
import scipy
import scipy.io as sio
import scipy.misc
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import json
import os
import os.path
from tqdm import tqdm

#####
#create the train and val obgt

def create_odgt(root_dir, file_dir, ann_dir, out_dir, anom_files=None):
    if anom_files is  None:
        anom_files = []
    _files = []

    count = total = 0
    print(root_dir)
    print(file_dir)
    town_names = sorted(os.listdir(root_dir+file_dir))
    print(town_names)

    for town in town_names:
        # img_files = sorted(os.listdir(os.path.join(root_dir,file_dir,town)))
        img_files = sorted([int(item[:-4]) for item in os.listdir(os.path.join(root_dir,file_dir,town))])
        img_files = [f"{item}.png" for item in img_files]
        #print(img_files)
        total += len(img_files)
        for img in img_files:
            ann_file = img
            ann_file_path = os.path.join(root_dir,ann_dir,town,ann_file)
            #print(ann_file_path)
            if os.path.exists(ann_file_path):
                dict_entry = {
                    "dbName": "StreetHazards",
                    "width": 1280,
                    "height": 720,
                    "fpath_img": os.path.join(file_dir,town,img),
                    "fpath_segm": os.path.join(ann_dir,town,ann_file),
                }
                count += 1
                _files.append(dict_entry)

    print("total images in = {} and out =  {}".format(total, count))

    with open(out_dir, "w") as outfile:
        json.dump(_files, outfile)

    return anom_files


### convert BDD100K semantic segmentation images to correct labels

def convert_bdd(root_dir, ann_dir):
    count = 0
    for img_loc in tqdm(os.listdir(root_dir+ann_dir)):
        img = Image.open(root_dir+ann_dir+img_loc)
        if img.ndim <= 1:
            continue
        #swap 255 with -1
        #16 -> 19
        #18 -> 16
        #19 -> 18
        # add 1 to whole array
        loc = img == 255
        img[loc] = -1
        loc = img == 16
        img[loc] = 19
        loc = img == 18
        img[loc] = 16
        loc = img == 19
        img[loc] = 18
        img += 1
        scipy.misc.toimage(img, cmin=0, cmax=255).save(root_dir+ann_dir+img_loc)



out_dir = "data/training.odgt"
#modify root directory to reflect the location of where the streethazards_train was extracted to.
root_dir = "data/"
train_dir = "train/images/training/"
ann_dir = "train/annotations/training/"
anom_files = create_odgt(root_dir, train_dir, ann_dir, out_dir)


out_dir = "data/validation.odgt"
train_dir = "train/images/validation/"
ann_dir = "train/annotations/validation/"
create_odgt(root_dir, train_dir, ann_dir, out_dir, anom_files=anom_files)


out_dir = "data/test.odgt"
val_dir = "test/images/test/"
ann_dir = "test/annotations/test/"
create_odgt(root_dir, val_dir, ann_dir, out_dir)

