import time
from tqdm import tqdm
from torchvision.transforms.functional import resize
from torchvision.transforms import InterpolationMode
import torch

class SegmentationModel(object):
    def __init__(self, args):
        self.remove_posthoc = args.remove_posthoc
        self.n_patches = args.n_patches
        self.location = args.location
        self.device = args.device

        # Initialize parameters to be filled in
        self.prompt = None
        self.original_image_shape = None
        self.post_hoc_mask = None
        self.train_time = None

    def segment(self, dataset):
        """
        Method that runs self.preprocess, then self.predict and self.postprocess for each datapoint.
        Uses self.preprocess and self.predict from subclasses.

        Parameters:
            - dataset: RiverSegmentationDataset object
        
        Returns:
            - list of dicts, one for each datapoint in the dataset
        """
        self.preprocess()

        # Predict for each datapoint
        output = []
        start = time.time()
        for data in tqdm(dataset):
            masks = self.predict(data)
            masks = self.postprocess(masks)
            data['predicted_masks'] = masks.cpu()
            output.append(data)
        
        end = time.time()
        self.test_time = round(end-start,3)
        return output

    def postprocess(self, masks: torch.Tensor):
        """
        Resizes predicted masks to original size and post-hoc removes masks out of bounds.

        Parameters:
            - masks : torch.Tensor, the masks to be processed  n_masks x h x w
        """
        # Resize masks to original size
        if masks.shape[0] > 0:
            masks = resize(masks, self.original_image_shape, interpolation=InterpolationMode.NEAREST)
        else:
            masks = torch.zeros(self.original_image_shape[-2:]) > 0

        # Posthoc remove out-of-bounds masks
        if not self.remove_posthoc:
            return masks
        masks = masks * self.post_hoc_mask.to(masks.device)
        return masks    
    
    def get_img_data(self, data : dict):
        """
        Returns image data in patches or fully, depending on settings
        """
        return data['patches'] if self.n_patches > 1 else data['images'].unsqueeze(0)

    def preprocess(self):
        """
        Initializes model, trains parameters if applicable (PerSAM-F, RandomForest), 
        prepares prompting images if necessary (PerSAM, SegGPT)
        """
        pass

    def predict(self):
        """
        For each datapoint, generates mask prediction.
        """
        pass


