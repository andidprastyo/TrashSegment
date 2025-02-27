'''Code adapted from https://christianjmills.com/posts/torchvision-cvat-annotation-tutorials/segmentation-polygons/
some code wrt bounding boxes was removed, so check the link for that
'''

from cjm_pytorch_utils.core import tensor_to_pil
import math
import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision.utils import draw_segmentation_masks

torchvision.disable_beta_transforms_warning()

from dataset import get_label_from_int

def visualize_example(output : dict, iou_summed_class, args):
    """
    Given an example output dictionary, visualizes image, ground truth and prediction.

    Arguments:
    output  :     dict containing 'images', 'predicted_masks', 'ground_truth_masks' and 'labels' keys
    n       :     int, number of examples to visualize per batch
    """
    # How many to visualize at a time
    n_per_plot = 5
    n_plots = (len(output) // n_per_plot) if args.n_to_visualize == -1 else args.n_to_visualize
    
    for n in range(n_plots):
        fig, axs = plt.subplots(n_per_plot, 3, squeeze=False, figsize=(10, 12))
        for i in range(n_per_plot):
            test_img = output[i+(n*n_per_plot)]['images']
            output_mask = output[i+(n*n_per_plot)]['predicted_masks']

            gt_masks = output[i+(n*n_per_plot)]['gt_masks_per_class'].squeeze()
            labels = output[i+(n*n_per_plot)]['labels'].squeeze()

            iou_insystem, _, iou_water, _ = iou_summed_class[i+(n*5)]
            show_images_gt_preds(test_img, output_mask, gt_masks, labels, axs[i], iou_insystem, iou_water)
        plt.suptitle('Original Image - Prediction - Ground Truth')
        plt.show()

def draw_ann(sample_img : torch.Tensor, masks: torch.Tensor, int_colors=None):
    """Draws given mask and labels on the given image
    
    Arguments: 
    sample_img  : image to show, 3 x H x W
    masks       : boolean mask, N x H x W
    int_colors  : colors to show, if None then it chooses a color itself

    Returns
    PIL Image   : image with segmentation mask on it, H x W x 3
    """
    annotated_tensor = draw_segmentation_masks(
        image=sample_img,
        masks=masks, 
        alpha=0.5, 
        colors=int_colors
    )
    result = tensor_to_pil(annotated_tensor)
    return result

def show_images_gt_preds(original : torch.Tensor, 
                        pred_mask : torch.Tensor, 
                        gt_mask : torch.Tensor,
                        labels : torch.Tensor, 
                        axs,
                        iou_insystem,
                        iou_water
                        ):
    """
    Visualizes original image, prediction and ground truth

    Arguments:
    original    : original image, 3 x H x W
    pred_mask   : boolean mask, M x H x W
    gt_mask     : boolean mask, N x H x W
    labels      : tensor of ints, size N
    axs         : axis to plot on
    show_title  : whether or not to show title of image
    """
    # Remove padding from labels and gt_masks
    labels = [get_label_from_int(l.item()) for l in labels]    
    if 'pad' in labels:
        cutoff = labels.index('pad')
        labels = labels[:cutoff]
        gt_mask = gt_mask[:cutoff]

    # Set colors for ground truth classes
    gt_colors = [[255, 177, 0],     # trash in system, yellow
            [241, 148, 180],        # trash outside system, pink
            [0, 108, 103],          # barrier, dark green
            [0, 56, 68],            # water, dark blue/green
            ]

    # Get colors for predicted masks
    pred_colors = [[255, 0, 110], [131, 56, 236], [251, 86, 7], [6, 214, 160]]
    pred_colors = pred_colors * math.ceil(len(pred_mask)/len(pred_colors))
    pred_colors = pred_colors[:len(pred_mask)]

    # Draw annotations
    prediction = draw_ann(original, pred_mask, pred_colors)
    ground_truth = draw_ann(original, gt_mask, gt_colors)

    title = f'IoU in system: {round(iou_insystem,3)}, water: {round(iou_water,3)}'
    plot_image(axs[0], original.permute(1,2,0))
    plot_image(axs[1], prediction, title=title)
    plot_image(axs[2], ground_truth)

def plot_image(ax, img, grid=None, title=''):
    """Utility method to plot an image without axes and with the given title"""
    ax.imshow(img)
    ax.axis('off')
    if grid is not None:
        ax.scatter(grid[:,0], grid[:,1], color='white', s=0.9)
    ax.title.set_text(title)
    ax.set_autoscale_on(False)
