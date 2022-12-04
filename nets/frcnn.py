import time

import numpy as np
import torch
import torch.nn as nn

from nets.classifier import Resnet50RoIHead, VGG16RoIHead,Resnet34RoIHead,Resnet101RoIHead
from nets.resnet import resnet50,resnet34,resnet101,resnet152
from nets.rpn import RegionProposalNetwork
import matplotlib.pyplot as plt

class FasterRCNN(nn.Module):
    def __init__(self, num_classes, 
                    mode = "training",
                    feat_stride = 16,
                    anchor_scales = [8, 16, 32],
                    ratios = [0.5, 1, 2],
                    backbone = 'vgg'):
        super(FasterRCNN, self).__init__()
        self.feat_stride = feat_stride
        if backbone == 'resnet50':
            self.extractor, classifier = resnet50()

            self.rpn = RegionProposalNetwork(
                1024, 512,
                ratios=ratios,
                anchor_scales=anchor_scales,
                feat_stride=self.feat_stride,
                mode = mode
            )
            self.head = Resnet50RoIHead(
                n_class=num_classes + 1,
                roi_size=14,
                spatial_scale=1,
                classifier=classifier
            )
        elif backbone == 'resnet101':
            self.extractor, classifier = resnet101()

            self.rpn = RegionProposalNetwork(
                1024, 512,
                ratios=ratios,
                anchor_scales=anchor_scales,
                feat_stride=self.feat_stride,
                mode = mode
            )
            self.head = Resnet101RoIHead(
                n_class=num_classes + 1,
                roi_size=14,
                spatial_scale=1,
                classifier=classifier
            )
        elif backbone == 'resnet152':
            self.extractor, classifier = resnet152()

            self.rpn = RegionProposalNetwork(
                1024, 512,
                ratios=ratios,
                anchor_scales=anchor_scales,
                feat_stride=self.feat_stride,
                mode = mode
            )
            self.head = Resnet50RoIHead(
                n_class=num_classes + 1,
                roi_size=14,
                spatial_scale=1,
                classifier=classifier
            )
        elif backbone == 'resnet34':
            self.extractor, classifier = resnet34()

            self.rpn = RegionProposalNetwork(
                256, 128,
                ratios=ratios,
                anchor_scales=anchor_scales,
                feat_stride=self.feat_stride,
                mode = mode
            )
            self.head = Resnet34RoIHead(
                n_class=num_classes + 1,
                roi_size=14,
                spatial_scale=1,
                classifier=classifier
            )
    def forward(self, x, scale=1.):
        img_size = x.shape[2:]
        base_feature = self.extractor(x)
        base_feature2=base_feature.detach().cpu().numpy()
        plt.matshow(base_feature2[0, 0, :, :], cmap='viridis')
        _, _, rois, roi_indices, _ = self.rpn(base_feature, img_size, scale)
        roi_cls_locs, roi_scores = self.head(base_feature, rois, roi_indices, img_size)
        return roi_cls_locs, roi_scores, rois, roi_indices

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
