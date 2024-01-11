# dataset.py

# custom segmentation dataset class which will load the images and masks, resize and normalize them. 

# INPUT
# Initialize class OxfordPetDataset(images_dir, mask_dir) with the directories for the images and the masks

# OUTPUT
# {image, mask} = dictionary. Keys: image, mask. Values: tensor, tensor

from os import listdir
from os.path import isfile, join, splitext
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm

def load_image(filename):
	# INPUT: file path to image
	# OUTPUT: PIL image object
    return Image.open(filename)

def unique_mask_values(idx, mask_dir):
	# function to find unique values in a given mask
	# commentary: is this necessary? Bc all masks have 0,1,2 as values I thought...

    # find the corresponding mask file
    mask_file = list(mask_dir.glob(idx + ".png"))[0]
    # convert the PIL Image to a numpy array
    mask = np.asarray(load_image(mask_file))
    # return grayscale mask unique values
    return np.unique(mask)

class OxfordPetDataset(Dataset):
    def __init__(self, images_dir: str, mask_dir: str):
        # initialize directories
        self.images_dir = Path(images_dir)
        self.mask_dir = Path(mask_dir)
        # generate a list of file ids
        self.ids = [
            splitext(file)[0]
            for file in listdir(images_dir)
            if isfile(join(images_dir, file)) and not file.startswith(".")
        ]
        # filter out 1 or 4 channel images
        self.ids = [
            img_id
            for img_id in self.ids
            if len(load_image(self.images_dir / f"{img_id}.jpg").split()) not in [1, 4]
        ]

        # throw an error if no images are found
        if not self.ids:
            raise RuntimeError(
                f"No input file found in {images_dir}, make sure you put your images there"
            )
        # print the number of examples
        print(f"[INFO] Creating dataset with {len(self.ids)} examples")
        # find unique mask values across all images
        unique = []
        for img_id in tqdm(self.ids):
            unique_values = unique_mask_values(img_id, self.mask_dir)
            unique.append(unique_values)
        # sort and print the unique mask values
        self.mask_values = list(
            sorted(np.unique(np.concatenate(unique), axis=0).tolist())
        )
        print(f"[INFO] Unique mask values: {self.mask_values}")
	
    def __len__(self):
        # get the number of examples
        return len(self.ids)
    
	# decorator defines a method that does not belong to any instance of the class. Consequently, it cannot take self. parameter
    @staticmethod 
    def preprocess(mask_values, pil_img, is_mask):
        # resize image
        pil_img = pil_img.resize(
            (128, 128), resample=Image.NEAREST if is_mask else Image.BICUBIC
        )
        img = np.asarray(pil_img)
        # if it's a mask, remap unique values
        if is_mask:
            mask = np.zeros((128, 128), dtype=np.int64)
            for i, v in enumerate(mask_values):
                mask[img == v] = i
            return mask
        else:
            # rearrange dimensions from width, height, channels to CWH
            img = img.transpose((2, 0, 1))
            # normalize the image
            if (img > 1).any():
                img = img / 255.0
            return img

    # get an example using an index
    def __getitem__(self, idx):
        # get the id using index
        name = self.ids[idx]
        # find the corresponding mask and image files
        mask_file = list(self.mask_dir.glob(name + ".png"))
        img_file = list(self.images_dir.glob(name + ".jpg"))
        # load the image and mask
        mask = load_image(mask_file[0])
        img = load_image(img_file[0])
        # check if the dimensions match
        assert (
            img.size == mask.size
        ), f"Image and mask {name} should be the same size, but are {img.size} and {mask.size}"
        # preprocess the image and mask
        img = self.preprocess(self.mask_values, img, is_mask=False)
        mask = self.preprocess(self.mask_values, mask, is_mask=True)
        # return as pytorch tensors
        return {
            "image": torch.as_tensor(img.copy()).float().contiguous(),
            "mask": torch.as_tensor(mask.copy()).long().contiguous(),
        }
