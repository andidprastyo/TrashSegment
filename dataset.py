import json
import numpy as np
import os
from pycocotools.coco import COCO
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms.functional import resize

class RiverSegmentationDataset(Dataset):
    """
    Custom class for River data.
    """
    def __init__(self, args):
        self.location = args.location
        self.n_patches = args.n_patches
        self.timeseries = args.timeseries

        self.image_folder = f'{args.images_path}/loc{args.location}'
        self.images = os.listdir(self.image_folder) 
        
        self.annotations = get_gt_masks(args.annotation_path, self.image_folder)
        
        # Select images for prompting
        all_prompt_imgs = list(prompt_images[args.location].keys())
        self.prompt_imgs = [all_prompt_imgs[i] for i in args.prompt_imgs]
        self.prompt_data = self.get_prompt_data(args) if args.model != 'YOLO' else None

        # Remove prompt images from dataset
        for img in self.prompt_imgs:
            self.images.remove(img)

        # Use YOLO images if applicable
        if args.yolo_test_path is not None:
            self.images = get_yolo_images(args.yolo_test_path, self.location)

        # Use timeseries data if applicable
        if self.timeseries is not None:
            self.images = os.listdir(self.timeseries)
            self.image_folder = self.timeseries

    def __len__(self):
        return len(self.images)
    
    def process_image(self, img_id, return_only_image=False):
        img_path = os.path.join(self.image_folder, img_id)
        image = read_image(img_path)

        # Make sure images are all the same size
        image = resize(image, (1944, 2592))

        # Split image into patches
        patches = torch.zeros(1,1)
        if self.n_patches > 1:
            _, h, w = image.shape
            h_patch_size, w_patch_size = h // self.n_patches, w // self.n_patches
            patches = image.unfold(1, h_patch_size, h_patch_size)
            patches = patches.unfold(2, w_patch_size, w_patch_size)
            patches = patches.permute(1,2,0,3,4)

            n_patches_h, n_patches_w, c, patch_h, patch_w = patches.shape
            patches = patches.reshape(n_patches_h*n_patches_w, c, patch_h, patch_w)

        if return_only_image:
            return {
                'images' : image,
                'patches' : patches,
                'img_id' : img_id
            }
        
        masks, labels = self.annotations[img_id]

        # Resize masks
        masks = resize(masks, (1944,2592)) 

        # Go from string to int labels
        labels = torch.Tensor([get_int_from_label(l) for l in labels])

        # Collect ground truth masks per class
        masks_per_class = [0]*(len(int_to_label.keys())-1)
        for l in int_to_label.keys():
            if l == -1:
                continue
            masks_per_class[l] = torch.sum(masks[labels==l], dim=0).unsqueeze(0)
        masks_per_class = torch.concat(masks_per_class)
        masks_per_class = masks_per_class > 0

        datapoint = {
            'images' : image,
            'patches' : patches,
            'gt_masks' : masks,
            'gt_masks_per_class' : masks_per_class,
            'labels' : labels,
            'img_id' : img_id
        }

        return datapoint
        
    def __getitem__(self, idx):
        return self.process_image(self.images[idx], return_only_image=self.timeseries is not None)
    
    def get_prompt_data(self, args):
        # Process all images and concatenate
        all_ref_imgs = []
        for img in self.prompt_imgs:
            data = self.process_image(img, return_only_image=False) 
            all_ref_imgs.append(data)
        data = river_collate(all_ref_imgs)

        # Select only the in-system ground truth masks and labels
        mask_indices = [prompt_images[args.location][x] for x in self.prompt_imgs]
        selected_gt_masks = []
        for i in range(len(self.prompt_imgs)):
            gt_masks = data['gt_masks'][i][data['labels'][i] == 0]
            selected_gt_masks.append(gt_masks[mask_indices[i]])
        selected_gt_masks = torch.concat([x.unsqueeze(0) for x in selected_gt_masks])

        data['gt_masks'] = selected_gt_masks > 0                  # n_prompt x 2 x H x W
        data['labels'] = torch.zeros(selected_gt_masks.shape)

        return data

def get_yolo_images(path_to_images, location):
    location_files = os.listdir(f'./data/images/loc{location}')
    split_files = os.listdir(path_to_images)
    current_images = set(location_files) & set(split_files)
    return list(current_images)

def river_collate(data):
    """Custom function to collate data for DataLoader"""
    labels = [x['labels'] for x in data]
    masks = [x['gt_masks'] for x in data]

    ## Compute the max number of masks in batch 
    max_n_masks = np.max([len(l) for l in labels])

    _, mask_h, mask_w = masks[0].shape

    ## Pad masks with zeros and labels with 'pad' label (-1 numerically)
    new_masks, new_labels = [], []
    for i, (m, l) in enumerate(zip(masks, labels)):
        l = l.tolist()
        n_masks = len(l)
        if n_masks != max_n_masks:
            to_add = max_n_masks-n_masks
            m = torch.concat([m, torch.zeros(to_add, mask_h, mask_w)])
            l += [-1 for _ in range(to_add)]
        new_masks.append(m > 0)    # ensure dtype bool 
        new_labels.append(torch.Tensor(l)) 

    ## Batch images, masks and labels
    image = torch.concat([i['images'].unsqueeze(0) for i in data])       # B x 3 x H x W 
    patches = torch.concat([i['patches'].unsqueeze(0) for i in data])
    masks_per_class = torch.concat([i['gt_masks_per_class'].unsqueeze(0) for i in data])    # B x 4 x H x W
    masks = torch.concat([m.unsqueeze(0) for m in new_masks])                               # B x n_masks x H x W
    labels = torch.concat([l.unsqueeze(0) for l in new_labels])                             # B x n_masks

    return {
        'images' : image,
        'patches' : patches,
        'gt_masks_per_class' : masks_per_class,
        'gt_masks' : masks,
        'labels' : labels
    }

def get_gt_masks(annotation_path, location_folder):
    # Load COCO annotation file
    coco = COCO(annotation_path)

    # Load classes
    catIds = coco.getCatIds()
    cats = coco.loadCats(catIds)
    cats = {x['id'] : x['name'] for x in cats}

    # Get images for current location
    imgIds = coco.getImgIds()
    location_img_ids = os.listdir(location_folder)
    all_imgs = coco.loadImgs(imgIds)
    imgs = [x for x in all_imgs if x['file_name'] in location_img_ids]
    
    # Collect ground truth masks and labels per image
    img_anns_labels = {}
    for img in imgs:
        anns = coco.loadAnns(coco.getAnnIds(imgIds=img['id'], catIds=catIds))
        labels = [cats[x['category_id']] for x in anns]
        masks = torch.concat([torch.from_numpy(coco.annToMask(x)).unsqueeze(0) for x in anns])
        
        img_anns_labels[img['file_name']] = (masks, labels)

    return img_anns_labels

## Mapping from label to int and from int to label
label_to_int = {
    'floating_trash_in_system' : 0,
    'floating_trash_outside_system' : 1,
    'water' : 2,
    'barrier' : 3,
    'pad' : -1
}
int_to_label = {}
for k,v in label_to_int.items():
    int_to_label[v] = k

def get_int_from_label(label):
    return label_to_int[label]

def get_label_from_int(int):
    return int_to_label[int]

prompt_images = {
    '1' : {
        '1702842300_20231217_194500_PTM5077.jpg' : [1, 3],
        '1702899900_20231218_114500_PTM5077.jpg' : [7, 8],
        '1698354900_20231026_211500_PTM5077.jpg' : [5, 7],
        '1699534800_20231109_130000_PTM5077.jpg' : [18, 20],
        '1702137600_20231209_160000_PTM5077.jpg' : [0, 2]
    },
    '2' : {
        '1702553400_20231214_113000_PTM5533.jpg' : [2, 6],
        '1703345400_20231223_153000_PTM5533.jpg' : [0, 3],
        '1703513700_20231225_141500_PTM5533.jpg' : [1, 2],
        '1704665700_20240107_221500_PTM5533.jpg' : [0, 3],
        '1705835700_20240121_111500_PTM5533.jpg' : [0, 2]
    },
    '3' : {
        '1705636800_20240119_040000_PTM5298.jpg' : [0, 1],
        '1705714200_20240120_013000_PTM5298.jpg' : [0, 5],
        '1707471000_20240209_093000_PTM5298.jpg' : [0, 1],
        '1705800600_20240121_013000_PTM5298.jpg' : [4, 6],
        '1706000400_20240123_090000_PTM5298.jpg' : [0, 1]
    }, 
    '4' : {
        'tl_ishem_ptm5116_2022-03-31T1120_1.jpg' : [0, 1],
        'tl_ishem_ptm5116_2022-04-21T0940_1.jpg' : [0, 1],
        'tl_ishem_ptm5116_2022-04-22T0640_1.jpg' : [2, 3],
        'tl_ishem_ptm5116_2022-04-17T1340_1.jpg' : [0, 1], 
        'tl_ishem_ptm5116_2022-04-22T0400_1.jpg' : [0, 1]
        },
    '5' : {
        '1693937400_20230905_181000_PTM5413.jpg' : [0, 1],
        '1693939200_20230905_184000_PTM5413.jpg' : [0, 1],
        '1693941600_20230905_192000_PTM5413.jpg' : [1, 2],
        '1700050800_20231115_122000_PTM5413.jpg' : [0, 1],
        '1700073600_20231115_184000_PTM5413.jpg' : [3, 4],
    },
    '6' : {
        '1705166100_20240113_171500_PTM5411.jpg' : [1, 3],
        '1705252500_20240114_171500_PTM5411.jpg' : [0, 1],
        '1705266900_20240114_211500_PTM5411.jpg' : [0, 3],
        '1705358700_20240115_224500_PTM5411.jpg' : [1, 2],
        '1707150600_20240205_163000_PTM5411.jpg' : [0, 1]
    }
}
