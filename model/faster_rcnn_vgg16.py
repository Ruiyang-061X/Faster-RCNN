from torch import nn
import numpy as np

from model.extractor_and_classifier import ExtractorVGG16, ClassifierVGG16
from model.rpn import RPN
from model.head import Head


class FasterRCNNVGG16(nn.Module):

    def __init__(self):
        super(FasterRCNNVGG16, self).__init__()
        self.location_normalize_mean = np.array([0.0, 0.0, 0.0, 0.0])
        self.location_normalize_std = np.array([0.1, 0.1, 0.2, 0.2])
        self.nms_thresh = 0.3
        self.score_thresh = 0.05
        self.extractor = ExtractorVGG16().cuda()
        self.classifier = ClassifierVGG16().cuda()
        self.rpn = RPN().cuda()
        self.head = Head(n_class=91, roi_size=7, spatial_scale=1.0 / 16, classifier=self.classifier).cuda()

    def forward(self, x, scale=1.0):
        image_size = x.shape[2 : ]
        x = self.extractor(x)
        rpn_location, rpn_score, roi, roi_indice, anchor = self.rpn(x, image_size, scale)
        roi_class_location, roi_class_score = self.head(x, roi, roi_indice)

        return roi_class_location, roi_class_score, roi, roi_indice


if __name__ == '__main__':
    import torch


    x = torch.rand((1, 3, 320, 480)).cuda()
    scale = 1.0

    faster_rcnn_vgg16 = FasterRCNNVGG16().cuda()

    faster_rcnn_vgg16(x, scale)