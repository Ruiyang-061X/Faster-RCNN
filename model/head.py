from collections import namedtuple
from string import Template

import torch
from torch.autograd import Function
from torch import nn
import cupy as cp

from model.utils.roi_cupy import kernel_forward, kernel_backward


Stream = namedtuple('Stream', ['ptr'])
CUDA_NUM_THREADS = 1024

@cp.util.memoize(for_each_device=True)
def load_kernel(kernel_name, code, **kwargs):
    cp.cuda.runtime.free(0)
    code = Template(code).substitute(**kwargs)
    kernel_code = cp.cuda.compile_with_cache(code)

    return kernel_code.get_function(kernel_name)

def get_blocks(N, K=CUDA_NUM_THREADS):

    return (N + K - 1) // K

class _RoIPooling2D(Function):

    def __init__(self, h, w, spatial_scale):
        self.h = h
        self.w = w
        self.spatial_scale = spatial_scale
        self.forward_function = load_kernel('roi_forward', kernel_forward)
        self.backward_function = load_kernel('roi_backward', kernel_backward)

    def forward(self, x, roi_indice_and_roi):
        x = x.contiguous()
        roi_indice_and_roi = roi_indice_and_roi.contiguous()
        _, c, h, w = x.size()
        n_ = roi_indice_and_roi.size(0)
        pooled_roi = torch.zeros(n_, c, self.h, self.w).cuda()
        self.x_size = x.size()
        self.n_ = roi_indice_and_roi.size(0)
        self.argmax_data = torch.zeros(n_, c, self.h, self.w).int().cuda()
        self.roi_indice_and_roi = roi_indice_and_roi
        args = [x.data_ptr(), roi_indice_and_roi.data_ptr(), pooled_roi.data_ptr(), self.argmax_data.data_ptr(), self.spatial_scale, c, h, w, self.h, self.w, pooled_roi.numel()]
        stream = Stream(ptr=torch.cuda.current_stream().cuda_stream)
        self.forward_function(args=args, block=(CUDA_NUM_THREADS, 1, 1), grid=(get_blocks(pooled_roi.numel()), 1, 1), stream=stream)

        return pooled_roi

    def backward(self, dpooled_roi):
        dpooled_roi = dpooled_roi.contiguous()
        dx = torch.zeros(self.x_size).cuda()
        _, c, h, w = self.x_size
        args = [dpooled_roi.data_ptr(), self.argmax_data.data_ptr(), self.roi.data_ptr(), dx.data_ptr(), self.n_, self.spatial_scale, c, h, w, self.h, self.w, dx.numel()]
        stream = Stream(ptr=torch.cuda.current_stream().cuda_stream)
        self.backward_function(args=args, block=(CUDA_NUM_THREADS, 1, 1), grid=(get_blocks(dx.numel()), 1,  1), stream=stream)

        return dx, None

class RoIPooling2D(nn.Module):

    def __init__(self, h, w, spatial_scale):
        super(RoIPooling2D, self).__init__()
        self._roi_pooling_2d = _RoIPooling2D(h, w, spatial_scale)

    def forward(self, x, roi_indice_and_roi):
        pooled_roi = self._roi_pooling_2d(x, roi_indice_and_roi)

        return pooled_roi

def normal_init(a, mean, std):
    a.weight.data.normal_(mean, std)
    a.bias.data.zero_()

class Head(nn.Module):

    def __init__(self, n_class, roi_size, spatial_scale, classifier):
        super(Head, self).__init__()
        self.n_class = n_class
        self.roi_size = roi_size
        self.spatial_scale = spatial_scale
        self.roi_pooling_2d = RoIPooling2D(roi_size, roi_size, spatial_scale)
        self.classifier = classifier
        self.class_location = nn.Linear(4096, n_class * 4)
        self.class_score = nn.Linear(4096, n_class)
        normal_init(self.class_location, 0, 0.001)
        normal_init(self.class_score, 0, 0.01)

    def forward(self, x, roi, roi_indice):
        roi = roi[0]
        roi_indice = roi_indice[0]
        roi = torch.from_numpy(roi).float().cuda()
        roi_indice = torch.from_numpy(roi_indice).float().cuda()
        roi_indice_and_roi = torch.cat([roi_indice[ : , None], roi], dim=1)[ : , [0, 2, 1, 4, 3]]
        pooled_roi = self.roi_pooling_2d(x, roi_indice_and_roi)
        pooled_roi = pooled_roi.view(pooled_roi.size(0), -1)
        out = self.classifier(pooled_roi)
        roi_class_location = self.class_location(out)
        roi_class_score = self.class_score(out)
        roi_class_location = roi_class_location.unsqueeze(0)
        roi_class_score = roi_class_score.unsqueeze(0)

        return roi_class_location, roi_class_score


if __name__ == '__main__':
    import numpy as np
    from model.extractor_and_classifier import ClassifierVGG16


    x = torch.rand((1, 512, 20, 30)).cuda()
    roi = np.random.rand(1, 1969, 4)
    roi_indice = np.random.rand(1, 1969)

    classifier = ClassifierVGG16().cuda()
    head = Head(n_class=91, roi_size=7, spatial_scale=1.0 /  16, classifier=classifier).cuda()

    head(x, roi, roi_indice)