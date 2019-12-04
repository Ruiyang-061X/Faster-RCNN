# Faster RCNN

## Introduction

This is a very short implementation of Faster RCNN using PyTorch.

## Usage

### Requirements

- Ubuntu 18.04
- Python==3.7.1
- torch==1.3.1
- torchvision==0.4.2
- tqdm==4.38.0
- numpy==1.17.4
- cupy-cuda101==6.5.0
- pycocotools==2.0
- Pillow==6.2.1
- six==1.13.0

### Installation

You should build the cython code in model/utils/nms/:

```Bash
cd model/utils/nms/
python build.py build_ext --inplace
```

### Dataset

You should prepare COCO dataset following the instructions in [COCO website](http://cocodataset.org/). You should also change the corresponding paths in the code.

### Train

You should locate at the root of this project and excute:

```Bash
python train.py
```

The trained models will be saved in checkpoints/.

### Performance

Due to lack of GPU resources, the model is left untrained.

This code is expected to get a mAP of around 30 on COCO testset.

## Citation

This work is mainly based on [chenyuntc's simple-faster-rcnn-pytorch](https://github.com/chenyuntc/simple-faster-rcnn-pytorch). Note that codes in model/utils/ are directly taken from simple-faster-rcnn-pytorch/model/utils.

The original paper where Faster RCNN comes from is [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](http://arxiv.org/pdf/1506.01497.pdf).