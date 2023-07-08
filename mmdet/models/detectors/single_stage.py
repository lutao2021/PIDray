# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch

from mmdet.core import bbox2result
from ..builder import DETECTORS, build_backbone, build_head, build_neck
from .base import BaseDetector

import torch.nn.functional as F

from ..confidence_branch.transformer import Transformer
from ..utils.positional_encoding import SinePositionalEncoding
import torch.nn as nn

@DETECTORS.register_module()
class SingleStageDetector(BaseDetector):
    """Base class for single-stage detectors.

    Single-stage detectors directly and densely predict bounding boxes on the
    output features of the backbone+neck.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(SingleStageDetector, self).__init__(init_cfg)
        if pretrained:
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            backbone.pretrained = pretrained
        self.backbone = build_backbone(backbone)
        if neck is not None:
            self.neck = build_neck(neck)
        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)
        self.bbox_head = build_head(bbox_head)
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

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/analysis_tools/get_flops.py`
        """
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        return outs

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):
        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_bboxes (list[Tensor]): Each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): Class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): Specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        ############zero padding#################
        h = img.shape[2]
        w = img.shape[3]
        if h < 320:
            img = F.pad(img,[0,0,0,320-h])
        if w < 320:
            img = F.pad(img,[0,320-w,0,0])
        ############zero padding#################

        super(SingleStageDetector, self).forward_train(img, img_metas)
        #####confidence branch#####
        x1 = self.extract_feat(img)
        target = torch.zeros(len(gt_bboxes), device=img.device)
        for i,label in enumerate(gt_bboxes):
            if len(label) > 0:
                target[i] = 1

        x0 = x1[-2]  # You can try to choose the other feature map to input to R0 for better performance.
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
        loss_confidence = 0.5 * loss_confidence

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
        else:
            x = x1
        #####confidence branch#####

        losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes,
                                              gt_labels, gt_bboxes_ignore)

        losses.update(dict(loss_confidence=loss_confidence))
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

    def simple_test(self, img, img_metas, rescale=False):
        """Test function without test-time augmentation.

        Args:
            img (torch.Tensor): Images with shape (N, C, H, W).
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        ############zero padding#################
        h = img.shape[2]
        w = img.shape[3]
        if h < 320:
            img = F.pad(img,[0,0,0,320-h])
        if w < 320:
            img = F.pad(img,[0,320-w,0,0])
        ############zero padding#################


        feat = self.extract_feat(img)
        x0 = feat[-2]

        results_list = self.bbox_head.simple_test(
            feat, img_metas, rescale=rescale)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in results_list
        ]

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
        confidence = self.confidence_transformer(x0.detach(), self.query_embed.weight, pos_embed)[0]
        confidence = self.confidence_fc(confidence[-1].squeeze(1))
        confidence_score = torch.sigmoid(confidence.detach())

        if confidence_score[0] <= 0.5:
            for i in range(len(bbox_results[0])):
                if bbox_results[0][i].shape[0] != 0:
                    for j in range(bbox_results[0][i].shape[0]):
                        bbox_results[0][i][j][4] = 0
        ########confidence branch##########

        return bbox_results

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test function with test time augmentation.

        Args:
            imgs (list[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch. each dict has image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        assert hasattr(self.bbox_head, 'aug_test'), \
            f'{self.bbox_head.__class__.__name__}' \
            ' does not support test-time augmentation'

        feats = self.extract_feats(imgs)
        results_list = self.bbox_head.aug_test(
            feats, img_metas, rescale=rescale)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in results_list
        ]
        return bbox_results

    def onnx_export(self, img, img_metas, with_nms=True):
        """Test function without test time augmentation.

        Args:
            img (torch.Tensor): input images.
            img_metas (list[dict]): List of image information.

        Returns:
            tuple[Tensor, Tensor]: dets of shape [N, num_det, 5]
                and class labels of shape [N, num_det].
        """
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        # get origin input shape to support onnx dynamic shape

        # get shape as tensor
        img_shape = torch._shape_as_tensor(img)[2:]
        img_metas[0]['img_shape_for_onnx'] = img_shape
        # get pad input shape to support onnx dynamic shape for exporting
        # `CornerNet` and `CentripetalNet`, which 'pad_shape' is used
        # for inference
        img_metas[0]['pad_shape_for_onnx'] = img_shape

        if len(outs) == 2:
            # add dummy score_factor
            outs = (*outs, None)
        # TODO Can we change to `get_bboxes` when `onnx_export` fail
        det_bboxes, det_labels = self.bbox_head.onnx_export(
            *outs, img_metas, with_nms=with_nms)

        return det_bboxes, det_labels
