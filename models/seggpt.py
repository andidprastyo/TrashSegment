import math
import torch
from transformers import SegGptImageProcessor, SegGptForImageSegmentation
from transformers import ViTImageProcessor, ViTForImageClassification

from segmentationModel import SegmentationModel

class SegGPTSegmenter(SegmentationModel):
    def __init__(self, args):
        super().__init__(args)
        self.n_patches = args.n_patches**2 if args.n_patches > 1 else 1
        self.bmc = args.binary_mask_cutoff
        self.match_prompt = args.match_prompt_img
    
    def preprocess(self):
        model_id = 'BAAI/seggpt-vit-large'
        self.processor = SegGptImageProcessor.from_pretrained(model_id)
        self.model = SegGptForImageSegmentation.from_pretrained(model_id).to(self.device)

        prompt_mask, prompt_image = self.prompt['gt_masks'], self.prompt['images']

        # Load embedder and get embeddings for prompts
        if self.match_prompt:
            model_id = 'google/vit-base-patch16-224'
            self.embed_processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
            self.embed_model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
            embed_inputs = self.embed_processor(images=prompt_image, return_tensors='pt')
            prompt_embeds = self.embed_model(**embed_inputs, output_hidden_states=True)
            self.prompt_embeds = prompt_embeds.hidden_states[-1]
    
        # Get prompt masks and images into correct format
        prompt_masks = prompt_mask.type(torch.uint8)*255
        prompt_masks = prompt_masks.unsqueeze(2).repeat([1,1,3,1,1])  # n_prompt x 2 x 3 x w x h

        if not self.match_prompt:
            prompt_image = torch.repeat_interleave(prompt_image, 2, 0)
            prompt_image = prompt_image.repeat([self.n_patches,1,1,1])

            prompt_masks = prompt_masks.reshape(2*len(prompt_masks), *prompt_masks.shape[2:])
            prompt_masks = prompt_masks.repeat([self.n_patches,1,1,1])

            self.prompt_masks, self.prompt_images = prompt_masks, prompt_image
        else:
            self.all_prompt_masks, self.all_prompt_images = prompt_masks, prompt_image

    def predict(self, data):
        # Get data in correct format
        img = self.get_img_data(data)
        self.original_image_shape = data['images'].shape[-2:]
        target_sizes = [list(img.shape[2:])]*2

        # Get image embedding
        if self.match_prompt:
            img_embed_input = self.embed_processor(images=img, return_tensors='pt')
            img_embeds = self.embed_model(**img_embed_input, output_hidden_states=True)
            img_embeds = img_embeds.hidden_states[-1]

            # Compute best matching prompt image
            similarities = torch.einsum('bij, bij -> b', img_embeds, self.prompt_embeds)
            best_match = torch.argmax(similarities).item()

            self.prompt_images = self.all_prompt_images[best_match].unsqueeze(0).repeat([2,1,1,1])
            self.prompt_masks = self.all_prompt_masks[best_match]
            image_input = img.repeat([2,1,1,1])
            masks = self.get_prediction(target_sizes, image_input, index=0)
            
        else:
            image_input = torch.repeat_interleave(img, 2*len(self.prompt['images']), dim=0)

            masks = []
            for i in range(0, len(image_input), 2):
                m = self.get_prediction(target_sizes, image_input, i) 
                masks.append(torch.concat([x.unsqueeze(0) for x in m]))
        masks = torch.concat([m.unsqueeze(0) for m in masks])

        # Correctly shape masks to combine into full images
        n_prompt_imgs, n_patches, h, w = len(self.prompt['images']), len(img), img.shape[-2], img.shape[-1]
        n_prompt_imgs = 1 if self.match_prompt else n_prompt_imgs
        masks = masks.reshape(n_patches, n_prompt_imgs*2, h, w)
        
        # Combine patches back into one image
        if self.n_patches > 1:
            masks = self.combine_patches(masks)
        masks = masks.squeeze()

        # Threshold and cutoff
        masks = torch.sigmoid(masks)
        masks = masks >= self.bmc
        return masks
    
    def combine_patches(self, patches):
        reshaped_patches = int(math.sqrt(self.n_patches))
        _, n_preds, h, w = patches.shape
        output_h, output_w = self.original_image_shape
        patches = patches.reshape(reshaped_patches, reshaped_patches, n_preds, h, w)
        patches = patches.permute(2,0,3,1,4).contiguous()
        patches = patches.view(n_preds, output_h, output_w)
        return patches
    
    def get_prediction(self, target_sizes, image_input, index):
        # Preprocess images and make prediction
        image_input, prompt_images, prompt_masks = [x[index:index+2] for x in 
                            [image_input, self.prompt_images, self.prompt_masks]]
        inputs = self.processor(
            images=image_input,
            prompt_images=prompt_images,
            prompt_masks=prompt_masks,
            return_tensors="pt",
        )
        for k,v in inputs.items():
            inputs[k] = v.to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Postprocess masks
        masks = self.processor.post_process_semantic_segmentation(outputs, target_sizes)
        return masks