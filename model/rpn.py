import numpy as np
from torch import nn
import torch.nn.functional as F

from model.utils.bbox_tools import generate_anchor_base
from model.utils.creator_tool import ProposalCreator


def normal_init(a, mean, std):
    a.weight.data.normal_(mean, std)
    a.bias.data.zero_()

def _enumerate_shifted_anchor(anchor_base, feature_stride, h, w):
    shift_y = np.arange(0, h * feature_stride, feature_stride)
    shift_x = np.arange(0, w * feature_stride, feature_stride)
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shift = np.stack((shift_y.ravel(), shift_x.ravel(), shift_y.ravel(), shift_x.ravel()), axis=1)
    A = anchor_base.shape[0]
    K = shift.shape[0]
    anchor = anchor_base.reshape((1, A, 4)) + shift.reshape((1, K, 4)).transpose((1, 0, 2))
    anchor = anchor.reshape((K * A, 4)).astype(np.float32)

    return anchor

class RPN(nn.Module):

    def __init__(self, in_channel=512, mid_channel=512, ratios=[0.5, 1, 2], anchor_scales=[4, 8, 16, 32], feature_stride=16, proposal_creator_parameter={}):
        super(RPN, self).__init__()
        self.anchor_base = generate_anchor_base(ratios=ratios, anchor_scales=anchor_scales)
        self.feature_stride = feature_stride
        self.conv = nn.Conv2d(in_channel, mid_channel, 3, 1, 1)
        n_anchor_base = self.anchor_base.shape[0]
        self.anchor_base_location = nn.Conv2d(mid_channel, n_anchor_base * 4, 1, 1, 0)
        self.anchor_base_score = nn.Conv2d(mid_channel, n_anchor_base * 2, 1, 1, 0)
        self.proposal_layer = ProposalCreator(self, **proposal_creator_parameter)
        normal_init(self.conv, 0, 0.01)
        normal_init(self.anchor_base_location, 0, 0.01)
        normal_init(self.anchor_base_score, 0, 0.01)

    def forward(self, x, image_size, scale=1.0):
        h, w = x.shape[2 : ]
        anchor = _enumerate_shifted_anchor(self.anchor_base, self.feature_stride, h, w)
        x = F.relu(self.conv(x))
        rpn_location = self.anchor_base_location(x)
        rpn_location = rpn_location.permute(0, 2, 3, 1).contiguous().view(-1, 4)
        rpn_score = self.anchor_base_score(x)
        rpn_score = rpn_score.permute(0, 2, 3, 1).contiguous().view(-1, 2)
        rpn_softmax_score = F.softmax(rpn_score, dim=1)
        rpn_foreground_score = rpn_softmax_score[ : , 1]
        roi = self.proposal_layer(rpn_location.cpu().detach().numpy(), rpn_foreground_score.cpu().detach().numpy(), anchor, image_size, scale)
        roi_indice = 0 * np.ones(roi.shape[0], dtype=np.int32)
        rpn_location = rpn_location.unsqueeze(0)
        rpn_score = rpn_score.unsqueeze(0)
        roi = np.array([roi])
        roi_indice = np.array([roi_indice])

        return rpn_location, rpn_score, roi, roi_indice, anchor


if __name__ == '__main__':
    import torch


    x = torch.rand((1, 512, 20, 30)).cuda()
    image_size = 320, 480
    scale = 0.8

    rpn = RPN().cuda()

    rpn(x, image_size, scale)