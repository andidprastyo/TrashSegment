from ultralytics import YOLO
from torchvision.transforms.functional import resize
import torch
import numpy as np

from segmentationModel import SegmentationModel

class YOLOSegmenter(SegmentationModel):
    def __init__(self, args):
        super().__init__(args)
        self.model_path = args.yolo_model_path

        # Set class names of interest
        self.target_class = 'floating_trash'
        self.barrier_class = 'barrier'

        # Set per location where to remove out of system predictions
        # axis to look at -- above or below to remove
        self.remove_barrier = {
        '1' : [0, 'above'],
        '2' : [0, 'above'],
        '3' : [0, 'below'],
        '4' : [0, 'above'],
        '5' : [0, 'below'],
        '6' : [1, 'below']
        }
    
    def preprocess(self):
        self.model = YOLO(self.model_path)

    def predict(self, data):
        # Get data in correct format
        img = self.get_img_data(data)
        self.original_image_shape = img.shape[-2:]
        input = resize(img, (960,1280)).float()

        # Make prediction
        prediction = self.model.predict(input,show_conf=False,show_boxes=False,show_labels=True)[0]
    
        # Check if masks have been predicted
        if prediction.masks is None:
            masks = torch.zeros(self.original_image_shape[-2:]) > 0
        else:
            masks = prediction.masks.data > 0

        # Check which class ids represents trash and barrier
        class_names = prediction.names
        trash_class_id, barrier_id = None, None
        for k, v in class_names.items():
            if v == self.target_class:
                trash_class_id = k
            if v == self.barrier_class:
                barrier_id = k
        assert trash_class_id is not None

        # If nothing is predicted, return zeros
        if len(prediction.boxes.cls) == 0:
            return torch.zeros(self.original_image_shape).unsqueeze(0) > 0

        # Select only trash masks
        predicted_classes = prediction.boxes.cls
        masks = masks
        trash_masks = masks[predicted_classes == trash_class_id].cpu()

        ## Remove out-system predictions if a barrier is predicted
        if barrier_id is not None:

            # Locate barrier
            barrier_pred = masks[predicted_classes == barrier_id].cpu()
            barrier_pred = torch.sum(barrier_pred, dim=0) > 0
            barrier_center = [np.average(indices) for indices in np.where(barrier_pred >= 1)]

            # Remove predictions
            for i, mask in enumerate(trash_masks):
                mask_center = [np.average(indices) for indices in np.where(mask >= 1)]
                axis, above_below = self.remove_barrier[self.location]
                if above_below == 'below':
                    if barrier_center[axis] < mask_center[axis]:
                        trash_masks[i] = torch.zeros(mask.shape)
                else:
                    if barrier_center[axis] > mask_center[axis]:
                        trash_masks[i] = torch.zeros(mask.shape)
        return trash_masks
    