from models.unet import UNet

class UNetSegmenter(SegmentationModel):
    def __init__(self, args):
        super().__init__(args)
        self.model = UNet(in_channels=3, out_channels=4)  # 4 classes: trash_in_system, trash_out_system, water, barrier
        self.model.to(self.device)
        self.model.load_state_dict(torch.load(args.unet_model_path))  # Load pretrained weights

    def preprocess(self):
        # No preprocessing needed for UNet
        pass

    def predict(self, data):
        img = self.get_img_data(data)
        img = img.to(self.device)
        with torch.no_grad():
            output = self.model(img)
        return output

    def postprocess(self, masks):
        # Resize masks to original size
        masks = F.interpolate(masks, size=self.original_image_shape, mode='bilinear', align_corners=False)
        masks = masks > 0.5  # Convert probabilities to binary masks
        return masks