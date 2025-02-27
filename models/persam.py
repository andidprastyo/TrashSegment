import numpy as np
import time
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.transforms.functional import resize, to_pil_image
from tqdm import tqdm
from transformers import AutoProcessor, SamModel
from typing import Tuple

from segmentationModel import SegmentationModel

class PerSAMSegmenter(SegmentationModel):
    def __init__(self, args):
        super().__init__(args)
        self.training = args.persam_training
        self.min_feature_sim = args.min_feature_sim

    def preprocess(self):
        # Initialize model
        self.processor = AutoProcessor.from_pretrained('facebook/sam-vit-huge')
        self.model = SamModel.from_pretrained('facebook/sam-vit-huge').to(self.device)

        # Prepare inputs for PerSAM or PerSAM-F
        self.prompt_mask, self.prompt_image = self.prompt['gt_masks'], self.prompt['images']
        prepped_inputs = self.get_target_feat(self.prompt_image, self.prompt_mask)

        # Finetune parameters for PerSAM-F
        start = time.time()
        self.weights = self.finetune_model(prepped_inputs['inputs']) if self.training else None
        self.train_time= round(time.time() - start, 3)

        self.target_feat, self.target_embedding = prepped_inputs['target_feat'], prepped_inputs['target_embeddings']

    def predict(self, data):
        img = self.get_img_data(data)
        self.original_image_shape = img.shape[-2:]

        # Get test image embeddings
        inputs = self.processor(images=img, return_tensors='pt').to(self.device)
        pixel_values = inputs.pixel_values
        with torch.no_grad():
            test_embeddings = self.model.get_image_embeddings(pixel_values)
            test_feat = test_embeddings.squeeze()

        # Cosine similarity
        num_channels, h, w = test_feat.shape
        test_feat = test_feat / test_feat.norm(dim=0, keepdim=True)
        test_feat_reshaped = test_feat.reshape(num_channels, h * w)
        sim = self.get_cosine_sim(self.target_feat, test_feat_reshaped, h, w, inputs)

        pred_masks, feature_sims, sims_to_plot = [], [], []
        sims_to_plot.append(sim)
        feature_sims.append(torch.mean(sim).item())
        while torch.mean(sim).item() > self.min_feature_sim:
            # Get prediction and update similarity
            if self.training:
                mask = self.get_prediction_trained(img, sim, test_embeddings.squeeze())
            else:
                mask = self.get_prediction_untrained(self.target_embedding, img, test_embeddings.squeeze(), sim)
            sim = sim.cpu() * (~mask.cpu())
            sims_to_plot.append(sim)

            # Save prediction and similarity
            pred_masks.append(mask.unsqueeze(0))
            feature_sims.append(torch.mean(sim).item())

            # Check if stopping conditions have been met
            if len(feature_sims) > 2 and feature_sims[-2] - feature_sims[-1] < 1e-4:
                feature_sims = feature_sims[:-1]
                pred_masks = pred_masks[:-1]
                sims_to_plot = sims_to_plot[:-1]
                break 
            sim = sim.to(self.device)
        
        # # Uncomment to plot the feature similarities
        # fig, axs = plt.subplots(len(sims_to_plot),2)
        # for i in range(len(sims_to_plot)):
        #     axs[i,0].imshow(to_pil_image(comp_imgs.squeeze()))
        #     axs[i,1].imshow(sims_to_plot[i].cpu())
        #     axs[i,1].set_title(round(feature_sims[i],3))
        #     [ax.axis('off') for ax in axs[i]]
        # plt.show()

        masks = torch.concat(pred_masks) > 0 if len(pred_masks) > 0 else torch.zeros(sim.shape).unsqueeze(0)
        return masks

    def get_prediction_untrained(self, target_embedding, test_image, test_embedding, sim):
        # Positive-negative location prior
        topk_xy, topk_label, last_xy_i, last_label_i = self.point_selection(sim, topk=1)
    
        topk_xy = np.concatenate([topk_xy, last_xy_i], axis=0)
        topk_label = np.concatenate([topk_label, last_label_i], axis=0)

        # Obtain the target guidance for cross-attention layers
        sim = (sim - sim.mean()) / torch.std(sim)
        sim = F.interpolate(sim.unsqueeze(0).unsqueeze(0), size=(64, 64), mode="bilinear")
        attention_similarity = sim.sigmoid_().unsqueeze(0).flatten(3)
        multimask_output = False

        # Initial prediction
        inputs = self.processor(test_image, input_points=[topk_xy.tolist()], input_labels=[topk_label.tolist()], return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(
                input_points=inputs.input_points, 
                input_labels=inputs.input_labels,
                image_embeddings=test_embedding.unsqueeze(0),
                multimask_output=multimask_output,
                attention_similarity=attention_similarity,  # Target-guided Attention
                target_embedding=target_embedding  # Target-semantic Prompting
            )
        best_idx = 0
        input_masks = outputs.pred_masks.squeeze(1)[best_idx: best_idx + 1, :, :]
        
        # Cascaded post-refinement 1
        with torch.no_grad():
            outputs_1 = self.model(
                input_points=inputs.input_points,
                input_labels=inputs.input_labels,
                input_masks=input_masks,
                image_embeddings=test_embedding.unsqueeze(0),
                multimask_output=True) 

        # Cascaded post-refinement 2
        masks = self.processor.image_processor.post_process_masks(outputs_1.pred_masks.cpu(),
                                                            inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu())[0].squeeze().numpy()
        best_idx = torch.argmax(outputs_1.iou_scores).item()
        input_box = self.get_input_box(masks[best_idx])

        # Parameters post-refinement 2
        inputs = self.processor(test_image, input_points=[topk_xy.tolist()], input_labels=[topk_label.tolist()], input_boxes=input_box,
                return_tensors="pt").to(self.device)
        input_boxes = inputs.input_boxes
        
        final_outputs = self.model(
            input_points=inputs.input_points, 
            input_labels=inputs.input_labels,
            input_boxes=input_boxes,
            input_masks=outputs_1.pred_masks.squeeze(1)[:,best_idx: best_idx + 1, :, :], 
            image_embeddings=test_embedding.unsqueeze(0),
            multimask_output=True)
        
        best_idx = torch.argmax(final_outputs.iou_scores).cpu()
        masks = self.processor.image_processor.post_process_masks(final_outputs.pred_masks.cpu(),
                    inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu())[0].squeeze()
        
        mask = masks[best_idx]
        return mask

    def get_prediction_trained(self, test_image, test_feature_sim, test_image_embeddings):
        # Positive location prior
        topk_xy, topk_label, _, _ = self.point_selection(test_feature_sim, topk=1)  
        inputs = self.processor(test_image, input_points=[topk_xy.tolist()], input_labels=[topk_label.tolist()], return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(
                input_points=inputs.input_points,
                input_labels=inputs.input_labels,
                image_embeddings=test_image_embeddings.unsqueeze(0),
                multimask_output=True,
            )

        weights_np = self.weights.detach().cpu().numpy()
        logits = outputs.pred_masks[0].squeeze(0).detach().cpu().numpy()
        logits = logits * weights_np[..., None]
        logit = logits.sum(0)  

        # Weighted sum three-scale masks
        logits_high = logits_high = self.postprocess_masks(masks=outputs.pred_masks.squeeze(1),
                                            input_size=inputs.reshaped_input_sizes[0].tolist(),
                                            original_size=inputs.original_sizes[0].tolist())
        logits_high = logits_high[0] * self.weights.unsqueeze(-1)
        logit_high = logits_high.sum(0)
        mask = (logit_high > 0).detach().cpu().numpy()

        input_box = self.get_input_box(mask)
        input_boxes = self.processor(test_image, input_boxes=[input_box], return_tensors="pt").input_boxes.to(self.device)

        # Cascaded Post-refinement-1
        with torch.no_grad():
            outputs_1 = self.model(
                        input_points=inputs.input_points,
                        input_labels=inputs.input_labels,
                        input_boxes=input_boxes,
                        input_masks=torch.tensor(logit[None, None, :, :], device=self.device),
                        image_embeddings=test_image_embeddings.unsqueeze(0),
                        multimask_output=True)    

        # Cascaded Post-refinement-2
        masks = self.processor.image_processor.post_process_masks(outputs_1.pred_masks.cpu(),
                                                            inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu())[0].squeeze().numpy()
        best_idx = torch.argmax(outputs_1.iou_scores).item()
        input_box = self.get_input_box(masks[best_idx])

        input_boxes = self.processor(test_image, input_boxes=[input_box], return_tensors="pt").input_boxes.to(self.device)

        final_outputs = self.model(
            input_points=inputs.input_points, 
            input_labels=inputs.input_labels,
            input_boxes=input_boxes,
            input_masks=outputs_1.pred_masks.squeeze(1)[:,best_idx: best_idx + 1, :, :], 
            image_embeddings=test_image_embeddings.unsqueeze(0),
            multimask_output=True)
        
        best_idx = torch.argmax(final_outputs.iou_scores).cpu()
        masks = self.processor.image_processor.post_process_masks(final_outputs.pred_masks.cpu(),
                    inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu())[0].squeeze()
        
        mask = masks[best_idx]
        return mask

    def finetune_model(self, all_inputs):
        # Learnable mask weights
        mask_weights = Mask_Weights()
        mask_weights.to(self.device)
        mask_weights.train()

        num_epochs = 1000
        log_epoch = 200
        optimizer = AdamW(mask_weights.parameters(), lr=1e-3, eps=1e-4)
        scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=num_epochs)  

        for train_idx in tqdm(range(num_epochs)):
            for inputs, gt_mask, image_embedding in all_inputs:
                # Run the decoder
                with torch.no_grad():
                    outputs = self.model(
                        input_points=inputs.input_points,
                        input_labels=inputs.input_labels,
                        image_embeddings=image_embedding.to(self.device),
                        multimask_output=True,
                    )

                logits_high = self.postprocess_masks(masks=outputs.pred_masks.squeeze(1),
                                                input_size=inputs.reshaped_input_sizes[0].tolist(),
                                                original_size=inputs.original_sizes[0].tolist())
                logits_high = logits_high[0].flatten(1)

                # Weighted sum three-scale masks
                weights = torch.cat((1 - mask_weights.weights.sum(0).unsqueeze(0), mask_weights.weights), dim=0)
                logits_high = logits_high * weights
                logits_high = logits_high.sum(0).unsqueeze(0)

                dice_loss = self.calculate_dice_loss(logits_high, gt_mask)
                focal_loss = self.calculate_sigmoid_focal_loss(logits_high, gt_mask)
                loss = dice_loss + focal_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                # if train_idx % log_epoch == 0:
                #     print('Train Epoch: {:} / {:}'.format(train_idx, num_epochs))
                #     current_lr = scheduler.get_last_lr()[0]
                #     print('LR: {:.6f}, Dice_Loss: {:.4f}, Focal_Loss: {:.4f}'.format(current_lr, dice_loss.item(), focal_loss.item()))
        mask_weights.eval()
        weights = torch.cat((1 - mask_weights.weights.sum(0).unsqueeze(0), mask_weights.weights), dim=0)
        return weights

    def get_target_feat(self, images, masks, concat_target=True):
        images = torch.repeat_interleave(images, repeats=2, dim=0)
        masks = masks.reshape(images.shape[0], masks.shape[-2], masks.shape[-1])

        all_target_feat, all_target_embed, all_inputs = [], [], []
        for i in range(len(masks)):
            ref_mask, ref_image = masks[i], images[i]

            # Get prompt and ground truth masks in correct format
            ref_mask = ref_mask.unsqueeze(2).repeat([1,1,3]).float().numpy()
            gt_mask = torch.tensor(ref_mask)[:, :, 0] > 0
            gt_mask = gt_mask.float().unsqueeze(0).flatten(1)  

            # Step 1: Image features encoding
            inputs =  self.processor(images=ref_image, return_tensors="pt").to(self.device)
            pixel_values = inputs.pixel_values

            with torch.no_grad():
                image_embeddings = self.model.get_image_embeddings(pixel_values.to(self.device))
                ref_feat = image_embeddings.squeeze().permute(1, 2, 0)

            # Step 2: interpolate prompt mask
            ref_mask = self.prepare_mask(ref_mask)
            ref_mask = F.interpolate(ref_mask, size=ref_feat.shape[0: 2], mode="bilinear")
            ref_mask = ref_mask.squeeze()[0]

            if torch.sum(ref_mask > 0) == 0:
                raise ValueError('Ref mask has no true values')

            # Step 3: Target feature extraction
            target_feat = ref_feat[ref_mask > 0]
            target_embedding = target_feat.mean(0).unsqueeze(0)
            target_embedding = target_embedding.unsqueeze(0)
            
            if self.training:
                target_feat_mean = target_feat.mean(0)
                target_feat_max = torch.max(target_feat, dim=0)[0]
                target_feat = (target_feat_max / 2 + target_feat_mean / 2).unsqueeze(0)  

                # Target and feature cosine similarity
                h, w, C = ref_feat.shape
                target_feat = target_feat / target_feat.norm(dim=-1, keepdim=True)
                ref_feat = ref_feat / ref_feat.norm(dim=-1, keepdim=True)
                ref_feat = ref_feat.permute(2, 0, 1).reshape(C, h * w)
                sim = self.get_cosine_sim(target_feat, ref_feat, h, w, inputs)

                # Positive location prior
                topk_xy, topk_label, _, _ = self.point_selection(sim, topk=1)

                inputs = self.processor(ref_image, input_points=[topk_xy.tolist()], input_labels=[topk_label.tolist()], return_tensors="pt").to(self.device) 
                gt_mask = gt_mask.to(self.device)
                all_inputs.append((inputs, gt_mask, image_embeddings))        
            else:
                target_feat = target_embedding / target_embedding.norm(dim=-1, keepdim=True)

            all_target_feat.append(target_feat.unsqueeze(0))
            all_target_embed.append(target_embedding.unsqueeze(0))
        
        target_feat = torch.mean(torch.concat(all_target_feat), dim=0) if concat_target else all_target_feat
        target_embed = torch.mean(torch.concat(all_target_embed), dim=0) if concat_target else all_target_embed
        return {'target_feat': target_feat, 
                'target_embeddings' : target_embed, 
                'inputs': all_inputs}  

    def get_cosine_sim(self, tensor1, tensor2, h, w, inputs):
        sim = tensor1 @ tensor2
        sim = sim.reshape(1, 1, h, w)
        sim = F.interpolate(sim, scale_factor=4, mode="bilinear")
        sim = self.processor.post_process_masks(sim.unsqueeze(1), original_sizes=inputs["original_sizes"].tolist(), reshaped_input_sizes=inputs["reshaped_input_sizes"].tolist(),
                                        binarize=False)
        sim = sim[0].squeeze()
        return sim

    def point_selection(self, mask_sim, topk=1):
        # Top-1 point selection
        w, h = mask_sim.shape
        topk_xy = mask_sim.flatten(0).topk(topk)[1]
        topk_x = (topk_xy // h).unsqueeze(0)
        topk_y = (topk_xy - topk_x * h)
        topk_xy = torch.cat((topk_y, topk_x), dim=0).permute(1, 0)
        topk_label = np.array([1] * topk)
        topk_xy = topk_xy.cpu().numpy()
            
        # Top-last point selection
        last_xy = mask_sim.flatten(0).topk(topk, largest=False)[1]
        last_x = (last_xy // h).unsqueeze(0)
        last_y = (last_xy - last_x * h)
        last_xy = torch.cat((last_y, last_x), dim=0).permute(1, 0)
        last_label = np.array([0] * topk)
        last_xy = last_xy.cpu().numpy()
        
        return topk_xy, topk_label, last_xy, last_label

    def prepare_mask(self, image, target_length=1024):
        target_size = self.get_preprocess_shape(image.shape[0], image.shape[1], target_length)
        mask = np.array(resize(to_pil_image(image), target_size))
        input_mask = torch.as_tensor(mask)
        input_mask = input_mask.permute(2, 0, 1).contiguous()[None, :, :, :]
        input_mask = self.preprocess_mask(input_mask)
        return input_mask

    def get_preprocess_shape(self, oldh: int, oldw: int, long_side_length: int) -> Tuple[int, int]:
        """
        Compute the output size given input size and target long side length.
        """
        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return (newh, neww)

    def preprocess_mask(self, x: torch.Tensor, pixel_mean=[123.675, 116.28, 103.53], pixel_std=[58.395, 57.12, 57.375], img_size=1024) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""

        pixel_mean = torch.Tensor(pixel_mean).view(-1, 1, 1)
        pixel_std = torch.Tensor(pixel_std).view(-1, 1, 1)

        # Normalize colors
        x = (x - pixel_mean) / pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = img_size - h
        padw = img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    def calculate_dice_loss(self, inputs, targets, num_masks = 1):
        """
        Compute the DICE loss, similar to generalized IOU for masks
        Args:
            inputs: A float tensor of arbitrary shape.
                    The predictions for each example.
            targets: A float tensor with the same shape as inputs. Stores the binary
                    classification label for each element in inputs
                    (0 for the negative class and 1 for the positive class).
        """
        inputs = inputs.sigmoid()
        inputs = inputs.flatten(1)
        numerator = 2 * (inputs * targets).sum(-1)
        denominator = inputs.sum(-1) + targets.sum(-1)
        loss = 1 - (numerator + 1) / (denominator + 1)
        return loss.sum() / num_masks

    def calculate_sigmoid_focal_loss(self, inputs, targets, num_masks = 1, alpha: float = 0.25, gamma: float = 2):
        """
        Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
        Args:
            inputs: A float tensor of arbitrary shape.
                    The predictions for each example.
            targets: A float tensor with the same shape as inputs. Stores the binary
                    classification label for each element in inputs
                    (0 for the negative class and 1 for the positive class).
            alpha: (optional) Weighting factor in range (0,1) to balance
                    positive vs negative examples. Default = -1 (no weighting).
            gamma: Exponent of the modulating factor (1 - p_t) to
                balance easy vs hard examples.
        Returns:
            Loss tensor
        """
        prob = inputs.sigmoid()
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p_t = prob * targets + (1 - prob) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** gamma)

        if alpha >= 0:
            alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
            loss = alpha_t * loss

        return loss.mean(1).sum() / num_masks

    def postprocess_masks(self, masks: torch.Tensor, input_size: Tuple[int, ...], original_size: Tuple[int, ...], img_size=1024) -> torch.Tensor:
        """
        Remove padding and upscale masks to the original image size.

        Arguments:
            masks (torch.Tensor): Batched masks from the mask_decoder,
            in BxCxHxW format.
            input_size (tuple(int, int)): The size of the image input to the
            model, in (H, W) format. Used to remove padding.
            original_size (tuple(int, int)): The original size of the image
            before resizing for input to the model, in (H, W) format.

        Returns:
            (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
            is given by original_size.
        """
        masks = F.interpolate(
            masks,
            (img_size, img_size),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : input_size[0], : input_size[1]]
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks  

    def get_input_box(self, mask):
        y, x = np.nonzero(mask)
        x_min = x.min()
        x_max = x.max()
        y_min = y.min()
        y_max = y.max()
        input_box = [[[x_min, y_min, x_max, y_max]]]
        return input_box
    
class Mask_Weights(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(2, 1, requires_grad=True) / 3)