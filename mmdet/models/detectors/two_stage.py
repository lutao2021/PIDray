# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import DETECTORS, build_backbone, build_head, build_neck
from .base import BaseDetector
from ..confidence_branch.transformer import Transformer
from ..utils.positional_encoding import SinePositionalEncoding
import numpy as np


@DETECTORS.register_module()
class TwoStageDetector(BaseDetector):
    """Base class for two-stage detectors.

    Two-stage detectors typically consisting of a region proposal network and a
    task-specific regression head.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 rpn_head=None,
                 roi_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(TwoStageDetector, self).__init__(init_cfg)
        if pretrained:
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            backbone.pretrained = pretrained
        self.backbone = build_backbone(backbone)

        if neck is not None:
            self.neck = build_neck(neck)

        if rpn_head is not None:
            rpn_train_cfg = train_cfg.rpn if train_cfg is not None else None
            rpn_head_ = rpn_head.copy()
            rpn_head_.update(train_cfg=rpn_train_cfg, test_cfg=test_cfg.rpn)
            self.rpn_head = build_head(rpn_head_)

        if roi_head is not None:
            # update train and test cfg here for now
            # TODO: refactor assigner & sampler
            rcnn_train_cfg = train_cfg.rcnn if train_cfg is not None else None
            roi_head.update(train_cfg=rcnn_train_cfg)
            roi_head.update(test_cfg=test_cfg.rcnn)
            roi_head.pretrained = pretrained
            self.roi_head = build_head(roi_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        #####confidence branch#####
        hidden_dim = 256
        self.confidence_transformer = Transformer(
                d_model=hidden_dim,
                dropout=0.1,
                nhead=4,
                dim_feedforward=hidden_dim*4,
                num_encoder_layers=0,
                num_decoder_layers=2,
                normalize_before=False,
                return_intermediate_dec=False,
                rm_self_attn_dec=True,
                rm_first_self_attn=True,
        )
        self.confidence_fc = nn.Linear(hidden_dim,1)
        self.query_embed = nn.Embedding(1, hidden_dim)
        self.positional_encoding = SinePositionalEncoding(num_feats=hidden_dim // 2, normalize=True)
        self.confidence_loss = nn.BCEWithLogitsLoss()
        #####confidence branch#####

    @property
    def with_rpn(self):
        """bool: whether the detector has RPN"""
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    @property
    def with_roi_head(self):
        """bool: whether the detector has a RoI head"""
        return hasattr(self, 'roi_head') and self.roi_head is not None

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        x0 = self.backbone(img)
        if self.with_neck:
            x = self.neck(x0)
        return x, x0

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/analysis_tools/get_flops.py`
        """
        outs = ()
        # backbone
        x, x0 = self.extract_feat(img)
        x0 = x[-1]  # You can try to choose the other feature map to input to R0 for better performance.
        # rpn
        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            outs = outs + (rpn_outs, )
        proposals = torch.randn(1000, 4).to(img.device)
        # roi_head
        roi_outs = self.roi_head.forward_dummy(x, proposals)

        ########confidence branch##########
        result = roi_outs
        batch_size = x0.size(0)
        input_img_h, input_img_w = x0.shape[-2:]
        masks = x0.new_ones((batch_size, input_img_h, input_img_w))
        for img_id in range(batch_size):
            img_h, img_w = x0.shape[-2:]
            masks[img_id, :img_h, :img_w] = 0

        # interpolate masks to have the same spatial shape with x
        masks = F.interpolate(
            masks.unsqueeze(1), size=x0.shape[-2:]).to(torch.bool).squeeze(1)
        # position encoding
        pos_embed = self.positional_encoding(masks)  # [bs, embed_dim, h, w]
        confidence = self.confidence_transformer(x0, self.query_embed.weight, pos_embed)[0]
        confidence = self.confidence_fc(confidence[-1].squeeze(1))
        confidence_score = torch.sigmoid(confidence.detach())

        if confidence_score[0] <= 0.5:
            if isinstance(result[0], tuple):
                for i in range(len(result[0][0])):
                    if result[0][0][i].shape[0] != 0:
                        for j in range(result[0][0][i].shape[0]):
                            result[0][0][i][j][4] = 0

            elif isinstance(result[0], list):
                for i in range(len(result[0])):
                    if result[0][i].shape[0] != 0:
                        for j in range(result[0][i].shape[0]):
                            result[0][i][j][4] = 0
        roi_outs = result
        ########confidence branch##########

        outs = outs + (roi_outs, )
        return outs

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):
        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.

            gt_labels (list[Tensor]): class indices corresponding to each box

            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

            proposals : override rpn proposals with custom proposals. Use when
                `with_rpn` is False.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        x1,x0 = self.extract_feat(img)

        losses = dict()

        #####confidence branch#####
        target = torch.zeros(len(gt_bboxes), device=img.device)
        for i,label in enumerate(gt_bboxes):
            if len(label) > 0:
                target[i] = 1
        super(TwoStageDetector, self).forward_train(img, img_metas)

        x0 = x1[-1]
        batch_size = x0.size(0)
        input_img_h, input_img_w = img_metas[0]['batch_input_shape']
        masks = x0.new_ones((batch_size, input_img_h, input_img_w))
        for img_id in range(batch_size):
            img_h, img_w, _ = img_metas[img_id]['img_shape']
            masks[img_id, :img_h, :img_w] = 0

        # interpolate masks to have the same spatial shape with x
        masks = F.interpolate(
            masks.unsqueeze(1), size=x0.shape[-2:]).to(torch.bool).squeeze(1)
        # position encoding
        pos_embed = self.positional_encoding(masks)  # [bs, embed_dim, h, w]
        confidence = self.confidence_transformer(x0, self.query_embed.weight, pos_embed)[0]
        confidence = self.confidence_fc(confidence[-1].squeeze(1))
        confidence_score = torch.sigmoid(confidence.detach())
        loss_confidence = self.confidence_loss(confidence.squeeze(1), target)
        loss_confidence = 0.1 * loss_confidence
        losses.update(dict(loss_confidence=loss_confidence))

        selected_id = torch.where(confidence_score[:,0] > 0.5)[0]
        if selected_id.shape[0] > 0:
            x = list()
            for i in range(len(x1)):
                x.append(x1[i][selected_id])
            x = tuple(x)
            img = img[selected_id]
            img_metas = [img_metas[selected_id[i]] for i in range(selected_id.shape[0])]
            gt_bboxes = [gt_bboxes[selected_id[i]] for i in range(selected_id.shape[0])]
            gt_labels = [gt_labels[selected_id[i]] for i in range(selected_id.shape[0])]
            if gt_masks:
                gt_masks = [gt_masks[selected_id[i]] for i in range(selected_id.shape[0])]
        else:
            x = x1

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                x,
                img_metas,
                gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg,
                **kwargs)
            losses.update(rpn_losses)
        else:
            proposal_list = proposals

        roi_losses = self.roi_head.forward_train(x, img_metas, proposal_list,
                                                 gt_bboxes, gt_labels,
                                                 gt_bboxes_ignore, gt_masks,
                                                 **kwargs)
        losses.update(roi_losses)

        if selected_id.shape[0] > 0:
            factor = float(selected_id.shape[0]) / x1[0].shape[0]
        else:
            factor = 0
        for loss_name, loss_value in losses.items():
            if 'confidence' not in loss_name:
                if isinstance(loss_value, torch.Tensor):
                    losses[loss_name] = loss_value.mean() * factor - (loss_value.mean() * factor).detach() + loss_value.mean().detach()
                elif isinstance(loss_value, list):
                    losses[loss_name] = sum(_loss.mean() * factor - (_loss.mean() * factor).detach() + _loss.mean().detach() for _loss in loss_value)
                else:
                    raise TypeError(
                        f'{loss_name} is not a tensor or list of tensors')

        return losses

    async def async_simple_test(self,
                                img,
                                img_meta,
                                proposals=None,
                                rescale=False):
        """Async test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'
        x = self.extract_feat(img)

        if proposals is None:
            proposal_list = await self.rpn_head.async_simple_test_rpn(
                x, img_meta)
        else:
            proposal_list = proposals

        return await self.roi_head.async_simple_test(
            x, proposal_list, img_meta, rescale=rescale)

    def simple_test(self, img, img_metas, proposals=None, rescale=False):
        """Test without augmentation."""

        assert self.with_bbox, 'Bbox head must be implemented.'
        x, x0 = self.extract_feat(img)
        x0 = x[-1]
        if proposals is None:
            proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
        else:
            proposal_list = proposals

        result =  self.roi_head.simple_test(
            x, proposal_list, img_metas, rescale=rescale)


        ########confidence branch##########
        batch_size = x0.size(0)
        input_img_h, input_img_w = img_metas[0]['batch_input_shape']
        masks = x0.new_ones((batch_size, input_img_h, input_img_w))
        for img_id in range(batch_size):
            img_h, img_w, _ = img_metas[img_id]['img_shape']
            masks[img_id, :img_h, :img_w] = 0

        # interpolate masks to have the same spatial shape with x
        masks = F.interpolate(
            masks.unsqueeze(1), size=x0.shape[-2:]).to(torch.bool).squeeze(1)
        # position encoding
        pos_embed = self.positional_encoding(masks)  # [bs, embed_dim, h, w]
        confidence = self.confidence_transformer(x0, self.query_embed.weight, pos_embed)[0]
        confidence = self.confidence_fc(confidence[-1].squeeze(1))
        confidence_score = torch.sigmoid(confidence.detach())

        if confidence_score[0] <= 0.5:
            if isinstance(result[0], tuple):
                for i in range(len(result[0][0])):
                    if result[0][0][i].shape[0] != 0:
                        for j in range(result[0][0][i].shape[0]):
                            result[0][0][i][j][4] = 0

            elif isinstance(result[0], list):
                for i in range(len(result[0])):
                    if result[0][i].shape[0] != 0:
                        for j in range(result[0][i].shape[0]):
                            result[0][i][j][4] = 0
        ########confidence branch##########

        return result

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        x = self.extract_feats(imgs)
        proposal_list = self.rpn_head.aug_test_rpn(x, img_metas)
        return self.roi_head.aug_test(
            x, proposal_list, img_metas, rescale=rescale)

    def onnx_export(self, img, img_metas):

        img_shape = torch._shape_as_tensor(img)[2:]
        img_metas[0]['img_shape_for_onnx'] = img_shape
        x = self.extract_feat(img)
        proposals = self.rpn_head.onnx_export(x, img_metas)
        if hasattr(self.roi_head, 'onnx_export'):
            return self.roi_head.onnx_export(x, proposals, img_metas)
        else:
            raise NotImplementedError(
                f'{self.__class__.__name__} can not '
                f'be exported to ONNX. Please refer to the '
                f'list of supported models,'
                f'https://mmdetection.readthedocs.io/en/latest/tutorials/pytorch2onnx.html#list-of-supported-models-exportable-to-onnx'  # noqa E501
            )
