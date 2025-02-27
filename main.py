import argparse
import ast
import numpy as np
import os
import random
import torch
from tqdm import tqdm

from visualization import visualize_example
from dataset import RiverSegmentationDataset

from models.yolo import YOLOSegmenter
from models.persam import PerSAMSegmenter
from models.seggpt import SegGPTSegmenter
from models.random_forest import RFSegmenter
from models.unet import UNetSegmenter

def main(args):
    # Set device
    set_seed()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.device = device
    print(f'Using device {args.device}')

    # Fix finetune imgs parameter
    if type(args.prompt_imgs) != list:
        args.prompt_imgs = ast.literal_eval(args.prompt_imgs)

    # When matching the finetune images to the current image, take all finetune imgs
    if args.match_prompt_img:
        args.prompt_imgs = [0,1,2,3,4]

    # Make sure CLIPSeg doesn't load a finetune image
    if args.model in ['CLIPSeg', 'YOLO']:
        args.prompt_imgs = []

    # Load dataset and posthoc mask
    test_dataset = RiverSegmentationDataset(args)
    print(f'Using {len(args.prompt_imgs)} images for prompting and {len(test_dataset)} images for testing.')
    args.post_hoc_mask = torch.load(f'./data/combined_gt_masks/loc{args.location}.pt').unsqueeze(0)

    # Make predictions
    segmenter = segment_call[args.model](args)
    if len(args.prompt_imgs) > 0:
        segmenter.prompt = test_dataset.prompt_data
    segmenter.post_hoc_mask = args.post_hoc_mask
    output = segmenter.segment(test_dataset)
    print(f'Finished predictions in {segmenter.test_time} seconds')

    # Stop if no ground truth data available
    if args.timeseries is not None:
        return output

    # Compute and log metrics
    metrics, visualization = get_metrics(output, args.location)
    metrics['test_time'] = segmenter.test_time
    metrics['train_time'] = segmenter.train_time
    
    # Visualize output
    iou_summed_class = visualization['summed_class_per_img']
    visualize_example(output, iou_summed_class, args)

    return output, visualization, metrics

def get_metrics(output, location):
    location_bins = {
    '1' : np.array([0.125, 0.25])*1e6,
    '2' : np.array([0.1, 0.2])*1e6,
    '3' : np.array([0.1, 0.2])*1e6,
    '4' : np.array([0.15, 0.3])*1e6,
    '5' : np.array([0.005, 0.02])*1e6,
    '6' : np.array([0.15, 0.3])*1e6,
    }

    iou_class, gt_mask_sizes, pred_mask_sizes, hamming_img = [], [], [], []
    hamming_distance = {'neg' : [], 'pos' : []}
    for batch in tqdm(output):
        gt_masks, gt_class = batch['gt_masks'], batch['gt_masks_per_class']
        labels, pred = batch['labels'], batch['predicted_masks']
        
        # Remove padding labels and masks
        if -1 in labels:
            non_padded = (labels != -1).nonzero(as_tuple=True)
            labels = labels[non_padded]
            gt_masks = gt_masks[non_padded]

        # Get IoU per class 
        combined_pred = (torch.sum(pred, dim=0) > 0).repeat([4,1,1])
        iou_class_summed = iou(gt_class, combined_pred)   
        iou_class.append(iou_class_summed.unsqueeze(0))

        # In-system gt masks
        in_system_masks =  gt_masks[np.where(np.array(labels) == 0)]
        gt_pixels = torch.sum(in_system_masks)

        # Hamming distance
        pred_pixels = torch.sum(combined_pred[0])
        hamming_dist = (gt_pixels - pred_pixels) / gt_pixels
        if gt_pixels != 0:
            hamming_img.append(hamming_dist.item())
            if hamming_dist < 0:
                hamming_distance['neg'].append(hamming_dist)
            else:
                hamming_distance['pos'].append(hamming_dist)

        # Collect gt and pred mask size
        gt_mask_sizes.append(gt_pixels.item())
        pred_mask_sizes.append(pred_pixels.item())

    # Fix what happens if the hamming distance is empty
    for setting, results in hamming_distance.items():
        if len(results) == 0:
            hamming_distance[setting] = [torch.zeros(1)]

    # Class iou to use for visualizations
    summed_class_per_image = torch.concat(iou_class)

    # Combine per-image metrics into general metrics
    iou_class = torch.concat(iou_class)
    mean_iou = [round(x.item(),4) for x in torch.mean(iou_class, dim=0)]
    std_iou = [round(x.item(),4) for x in torch.std(iou_class, dim=0)]
    mean_hamming_pos = round(torch.mean(torch.Tensor(hamming_distance['pos'])).item(),3)
    mean_hamming_neg = round(torch.mean(torch.Tensor(hamming_distance['neg'])).item(),3)

    # Compute IoU per bin
    gt_mask_sizes = np.array(gt_mask_sizes)
    in_system_iou = iou_class[:, 0]
    s_border, m_border = location_bins[location]
    iou_small = torch.mean(in_system_iou[np.where(gt_mask_sizes <= s_border)]).item()
    iou_medium = torch.mean(in_system_iou[np.where((gt_mask_sizes > s_border) & (gt_mask_sizes <= m_border))]).item()
    iou_large = torch.mean(in_system_iou[np.where(gt_mask_sizes > m_border)]).item()
    binned_iou = {'small': iou_small, 'medium': iou_medium, 'large': iou_large}

    metrics = {
        'iou_summed_class' : {
            'in_system' : mean_iou[0], 'in_system_std' : std_iou[0], 
            'out_system' : mean_iou[1], 'out_system_std' : std_iou[1],
            'water' : mean_iou[2], 'water_std': std_iou[2],
            'barrier' : mean_iou[3], 'barrier_std': std_iou[3]
            }, 
        'hamming' : {'pos': mean_hamming_pos, 'neg': mean_hamming_neg},
        'binned_iou' : binned_iou
        }
    visualization = {
        'gt_mask_sizes' : gt_mask_sizes,
        'pred_mask_sizes' : pred_mask_sizes,
        'summed_class_per_img' : summed_class_per_image.tolist(),
        'hamming_img' : hamming_img
    }
    return metrics, visualization

def iou(gt, pred):
    intersection = gt & pred
    union = gt | pred
    
    intersection = torch.sum(intersection, dim=[-1, -2])
    union = torch.sum(union, dim=[-1, -2])

    ious = intersection / (union + 1e-8)
    return ious

def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")

segment_call = {
    'PerSAM' : PerSAMSegmenter,
    'SegGPT' : SegGPTSegmenter,
    'RandomForest' : RFSegmenter,
    'YOLO' : YOLOSegmenter,
    'unet': UNetSegmenter
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # General settings
    parser.add_argument('--annotation_path', default='./data/annotations/all_annotations.json', help='Path to .json file containing annotations')
    parser.add_argument('--images_path', default='./data/images', help='Path to dataset')
    parser.add_argument('--n_to_visualize', default=0, type=int, help='Set to non-zero to visualize the first n batches (batch size = 5). For -1 it visualizes all batches')
    parser.add_argument('--timeseries', default=None, help='Add path to folder to predict on data without ground truth annotations. Bypasses visualization and metric computation.')

    # Experiment settings
    parser.add_argument('--model', default='SegGPT', choices=segment_call.keys(), help='Dictates which model to use, choices are "PerSAM", "SegGPT", "RandomForest", "YOLO", and "UNet')
    parser.add_argument('--location', default='1', choices=['1', '2', '3', '4', '5', '6'], type=str, help='Which location to use, number between 1 and 6 as a string')
    parser.add_argument('--remove_posthoc', action='store_true', default=True, help='Enable flag to remove irrelevant masks after prediction. These are masks that are outside of the area of interest (the river).')
    parser.add_argument('--prompt_imgs', default=[0], help='Which image(s) to use as prompts')
    parser.add_argument('--match_prompt_img', action='store_true', default=False, help='Enable to find the best prompt image for each testing image, using image embeddings.')
    parser.add_argument('--n_patches', type=int, default=1, help='Set to >1 to divide each axis of the image into patches. So --n_patches==3 makes 9 patches total')

    # SegGPT
    parser.add_argument('--binary_mask_cutoff', default=0.8, type=float, help='Dictates the cutoff for the logit output of SegGPT to convert to a binary mask.')

    # PerSAM
    parser.add_argument('--min_feature_sim', default=0.1, type=float, help='When mean feature similarity drops below this value, prediction of masks stops.')
    parser.add_argument('--persam_training', action='store_true', default=False, help='Use flag to use PerSAM-F instead of PerSAM.')

    # YOLO
    parser.add_argument('--yolo_test_path', default=None, help='Path to folder containing images for testing')
    parser.add_argument('--yolo_model_path', default='./models/yolo_seg_custom.pt', help='Path to pretrained YOLO model')
    
    # UNet
    parser.add_argument('--unet_model_path', default='./models/unet_model.pt', help='Path to pretrained U-Net model weights')
    parser.add_argument('--unet_input_height', type=int, default=512, help='Input height for U-Net model')
    parser.add_argument('--unet_input_width', type=int, default=512, help='Input width for U-Net model')

    args = parser.parse_args()
    assert not ((args.model == 'YOLO') and (args.yolo_test_path is None))

    main(args)