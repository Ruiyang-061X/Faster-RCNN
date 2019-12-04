import json

from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
import cupy as cp

from dataset import Coco
from torch.utils.data import DataLoader
from model.faster_rcnn_vgg16 import FasterRCNNVGG16
from model.utils.bbox_tools import loc2bbox
from model.utils.nms import non_maximum_suppression
from model.utils.creator_tool import AnchorTargetCreator, ProposalTargetCreator
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


trainset_root = '/media/ruiyang/Data/coco/images/train2017'
trainset_annFile = '/media/ruiyang/Data/coco/annotations/instances_train2017.json'
trainset = Coco(trainset_root, trainset_annFile)
validationset_root = '/media/ruiyang/Data/coco/images/val2017'
validationset_annFile = '/media/ruiyang/Data/coco/annotations/instances_val2017.json'
validationset = Coco(validationset_root, validationset_annFile)
trainset_loader = DataLoader(trainset)
validationset_loader = DataLoader(validationset)
faster_rcnn_vgg16 = FasterRCNNVGG16().cuda()

n_epoch = 14
print_every = 1000
lr = 1e-3
lr_decay = 1e-1
weight_decay = 5e-4
rpn_sigma = 3.0
roi_sigma = 1.0

parameters = []
for key, value in dict(faster_rcnn_vgg16.named_parameters()).items():
    if value[0].requires_grad:
        if 'bias' in key:
            parameters += [{'params': [value], 'lr': lr * 2, 'weight_decay': 0}]
        else:
            parameters += [{'params': [value], 'lr': lr, 'weight_decay': weight_decay}]
optimizer = Adam(parameters)
best_map = 0.0
best_path = ''


def suppress(roi_class_bbox, roi_softmax_class_score):
    predicted_bbox, predicted_label, predicted_score = [], [], []
    for c in range(faster_rcnn_vgg16.head.n_class):
        roi_class_bbox_c = roi_class_bbox[ : , c, : ]
        roi_softmax_class_score_c = roi_softmax_class_score[ : , c]
        mask = roi_softmax_class_score_c > faster_rcnn_vgg16.score_thresh
        roi_class_bbox_c = roi_class_bbox_c[mask]
        roi_softmax_class_score_c = roi_softmax_class_score_c[mask]
        keep = non_maximum_suppression(cp.array(roi_class_bbox_c), faster_rcnn_vgg16.nms_thresh, roi_softmax_class_score_c)
        keep = cp.asnumpy(keep)
        roi_class_bbox_c = roi_class_bbox_c[keep]
        roi_softmax_class_score_c = roi_softmax_class_score_c[keep]
        predicted_bbox += list(roi_class_bbox_c)
        predicted_label += roi_class_bbox_c.shape[0] * [c]
        predicted_score += list(roi_softmax_class_score_c)

    return predicted_bbox, predicted_label, predicted_score

def validation():
    faster_rcnn_vgg16.eval()
    image_ids, predicted_bboxes, predicted_labels, predicted_scores = [], [], [], []
    for i, (image_id, image, scale, _, _) in tqdm(enumerate(validationset_loader)):
        image_id = image_id[0]
        image_id = image_id.item()
        scale = scale[0]
        scale = scale.item()
        image = image.cuda()
        roi_class_location, roi_class_score, roi, _ = faster_rcnn_vgg16(image, scale)
        roi_class_location = roi_class_location[0]
        roi_class_score = roi_class_score[0]
        roi = roi[0]
        roi = torch.from_numpy(roi).cuda()
        mean = torch.from_numpy(faster_rcnn_vgg16.location_normalize_mean).cuda()
        std = torch.from_numpy(faster_rcnn_vgg16.location_normalize_std).cuda()
        roi_class_location = roi_class_location.view(-1, faster_rcnn_vgg16.head.n_class, 4)
        roi_class_location =  roi_class_location * std + mean
        roi = roi.view(-1, 1, 4).expand_as(roi_class_location)
        roi_class_location = roi_class_location.contiguous().view(-1, 4)
        roi = roi.contiguous().view(-1, 4)
        roi_class_bbox = loc2bbox(roi.cpu().detach().numpy(), roi_class_location.cpu().detach().numpy())
        roi_class_bbox = torch.from_numpy(roi_class_bbox).cuda()
        h, w = image.shape[2 : ]
        roi_class_bbox[ : , 0 : : 2] = roi_class_bbox[ : , 0 : : 2].clamp(min=0, max=h)
        roi_class_bbox[ : , 1 : : 2] = roi_class_bbox[ : , 1 : : 2].clamp(min=0, max=w)
        roi_softmax_class_score = F.softmax(roi_class_score, dim=1)
        roi_class_bbox = roi_class_bbox.view(-1, faster_rcnn_vgg16.head.n_class, 4)
        predicted_bbox, predicted_label, predicted_score = suppress(roi_class_bbox.cpu().detach().numpy(), roi_softmax_class_score.cpu().detach().numpy())
        image_ids += len(predicted_bbox) * [image_id]
        predicted_bboxes += predicted_bbox
        predicted_labels += predicted_label
        predicted_scores += predicted_score

    coco_ground_truth = COCO(validationset_annFile)
    validationset_resFile = './instances_val2017_bbox_results.json'
    predicted = []
    for i in range(len(predicted_bboxes)):
        dic = {'image_id' : image_ids[i], 'category_id' : predicted_labels[i], 'bbox' : list(predicted_bboxes[i].astype(float)), 'score' : float(predicted_scores[i])}
        predicted.append(dic)
    f = open(validationset_resFile, 'w')
    json.dump(predicted, f)
    f.close()
    coco_predicted = coco_ground_truth.loadRes(validationset_resFile)
    coco_validation = COCOeval(coco_ground_truth, coco_predicted, 'bbox')
    coco_validation.params.imgIds = image_ids
    coco_validation.evaluate()
    coco_validation.accumulate()
    coco_validation.summarize()
    map = coco_validation.stats[0]

    return map

def smooth_l1_loss(x, t, in_weight, sigma):
    sigma2  = sigma ** 2
    difference = in_weight * (x - t)
    abs_difference = difference.abs()
    flag = (abs_difference < (1.0 / sigma2)).float()
    y = flag * (sigma2 / 2.0) * (difference ** 2) + (1 - flag) * (abs_difference - 0.5 / sigma2)

    return y.sum()

def fast_rcnn_location_loss(predicted_location, ground_truth_location, ground_truth_label, sigma):
    in_weight = torch.zeros(ground_truth_location.size()).cuda()
    in_weight[(ground_truth_label > 0).view(-1, 1).expand_as(in_weight)] = 1
    location_loss = smooth_l1_loss(predicted_location, ground_truth_location, in_weight.detach(), sigma)
    location_loss /= (ground_truth_label >= 0).sum()

    return location_loss

for epoch in range(n_epoch):
    for i, (_, ground_truth_image, scale, ground_truth_bbox, ground_truth_label) in tqdm(enumerate(trainset_loader)):
        optimizer.zero_grad()
        ground_truth_image = ground_truth_image.cuda()
        ground_truth_bbox = ground_truth_bbox.cuda()
        ground_truth_label = ground_truth_label.long().cuda()
        ground_truth_bbox = ground_truth_bbox[0]
        ground_truth_label = ground_truth_label[0]
        scale = scale[0]
        scale = scale.item()
        x = faster_rcnn_vgg16.extractor(ground_truth_image)
        image_size = ground_truth_image.shape[2 : ]
        rpn_location, rpn_score, roi, roi_indice, anchor = faster_rcnn_vgg16.rpn(x, image_size, scale)
        rpn_location = rpn_location[0]
        rpn_score = rpn_score[0]
        roi = roi[0]
        anchor_target_creator = AnchorTargetCreator()
        ground_truth_rpn_location, ground_truth_rpn_label = anchor_target_creator(ground_truth_bbox.cpu().detach().numpy(), anchor, image_size)
        ground_truth_rpn_location = torch.from_numpy(ground_truth_rpn_location).cuda()
        ground_truth_rpn_label = torch.from_numpy(ground_truth_rpn_label).long().cuda()
        rpn_location_loss = fast_rcnn_location_loss(rpn_location, ground_truth_rpn_location, ground_truth_rpn_label, rpn_sigma)
        rpn_class_loss = F.cross_entropy(rpn_score, ground_truth_rpn_label, ignore_index=-1)
        proposal_target_creator = ProposalTargetCreator()
        sample_roi, ground_truth_sample_roi_location, ground_truth_sample_roi_label = proposal_target_creator(roi, ground_truth_bbox.cpu().detach().numpy(), ground_truth_label.cpu().detach().numpy(), faster_rcnn_vgg16.location_normalize_mean, faster_rcnn_vgg16.location_normalize_std)
        sample_roi = np.array([sample_roi])
        n_sample = sample_roi.shape[1]
        sample_roi_indice = 0 * np.ones((n_sample))
        sample_roi_indice = np.array([sample_roi_indice])
        sample_roi_class_location, sample_roi_class_score = faster_rcnn_vgg16.head(x, sample_roi, sample_roi_indice)
        sample_roi_class_location = sample_roi_class_location[0]
        sample_roi_class_score = sample_roi_class_score[0]
        sample_roi_class_location = sample_roi_class_location.view(n_sample, -1, 4)
        sample_roi_location = sample_roi_class_location[torch.arange(0, n_sample), torch.from_numpy(ground_truth_sample_roi_label)]
        ground_truth_sample_roi_location = torch.from_numpy(ground_truth_sample_roi_location).cuda()
        ground_truth_sample_roi_label = torch.from_numpy(ground_truth_sample_roi_label).long().cuda()
        roi_location_loss = fast_rcnn_location_loss(sample_roi_location, ground_truth_sample_roi_location, ground_truth_sample_roi_label, roi_sigma)
        roi_class_loss = F.cross_entropy(sample_roi_class_score, ground_truth_sample_roi_label)
        total_loss = rpn_location_loss + rpn_class_loss + roi_location_loss + roi_class_loss
        total_loss.backward()
        optimizer.step()
        if i % print_every == 0:
            print('iter: %d loss: %.4f' % (i, total_loss))

    map = validation()
    print('epoch: %d map: %.4f' % (epoch, map))
    if map > best_map:
        best_map = map
        path = 'checkpoints/faster_rcnn_%s' % (str(map))
        torch.save(faster_rcnn_vgg16.state_dict(), path)
        best_path = path
    if epoch == 9:
        faster_rcnn_vgg16.load_state_dict(torch.load(best_path))
        for param_group in optimizer.param_group:
            param_group['lr'] *= lr_decay