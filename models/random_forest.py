from sklearn.ensemble import RandomForestClassifier
import time
import torch

from segmentationModel import SegmentationModel

class RFSegmenter(SegmentationModel):
    def __init__(self, args):
        super().__init__(args)
        self.model = RandomForestClassifier(n_estimators=10)
    
    def preprocess(self):
        training_masks, training_images = self.prompt['gt_masks_per_class'], self.prompt['images']
        X, y = self.convert_to_input(training_masks, training_images)

        start = time.time()
        self.model = self.model.fit(X,y)
        end = time.time()
        self.training_time = round(end-start,3)

    def predict(self, data):
        img = self.get_img_data(data)
        self.original_image_shape = img.shape[-2:]
        img = img.squeeze()

        # Make prediction
        C, H, W = img.shape
        img = img.permute(1,2,0)
        X = img.reshape(H*W, C)
        pred = self.model.predict(X)
        pred = pred.reshape(H,W)

        # Select in_system and out_system predicted plastic
        # and combine into boolean mask
        in_system = pred == 1
        out_system = pred == 2
        pred = in_system + out_system

        pred = torch.from_numpy(pred).unsqueeze(0)
        return pred

    def convert_to_input(self, masks, images):
        # Combine class maps into one full segmentation map
        masks = masks.float()
        all_maps = []
        for current_mask in masks:
            full_segmentation_map = torch.zeros(current_mask.shape[1:])
            for i, m in enumerate(current_mask):
                # Guarantee no overlap -- 0 = nothing, 1 = in_system, 2 = out_system
                # 3 = water, 4 = barrier
                nonzero_result = torch.nonzero(full_segmentation_map, as_tuple=True)
                m[nonzero_result] = torch.zeros(len(nonzero_result[0]))

                full_segmentation_map += m*(i+1)
            all_maps.append(full_segmentation_map)
        masks = torch.concat([x.unsqueeze(0) for x in all_maps])

        B, C, H, W = images.shape
        B, H, W = masks.shape

        # Reshape images and masks to fit the desired input shape
        images = images.reshape(B, C, H*W)
        masks = masks.reshape(B, H*W)

        images = images.permute(0, 2, 1)
        images = images.reshape(B*H*W, C)
        masks = masks.reshape(B*H*W)
        return images, masks