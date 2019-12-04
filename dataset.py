import os

from torch.utils.data import Dataset
from pycocotools.coco import COCO
from PIL import Image
import numpy as np
from torchvision import transforms


class Coco(Dataset):

    def __init__(self, root, annFile):
        super(Coco, self).__init__()
        self.root = root
        self.coco = COCO(annFile)
        self.ids = sorted(self.coco.imgs.keys())

    def __getitem__(self, index):
        image_id = self.ids[index]
        ann_id = self.coco.getAnnIds(image_id)
        label = self.coco.loadAnns(ann_id)
        image_path = self.coco.loadImgs(image_id)[0]['file_name']
        image = Image.open(os.path.join(self.root, image_path)).convert('RGB')
        bbox = []
        label_ = []
        for dic in label:
            bbox.append(dic['bbox'])
            label_.append(dic['category_id'])
        bbox = np.array(bbox)
        label_ = np.array(label_)

        w, h = image.size
        min_size = 600
        max_size = 1000
        scale1 = min_size / min(h, w)
        scale2 = max_size / max(h, w)
        scale = min(scale1, scale2)
        image = transforms.Resize((int(h * scale), int(w * scale)))(image)
        image = transforms.ToTensor()(image)
        image = image / 255
        image = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image)
        bbox = bbox * scale

        return image_id, image, scale, bbox, label_

    def __len__(self):

        return len(self.ids)


if __name__ == '__main__':
    root = '/media/ruiyang/Data/coco/images/train2017'
    annFile = '/media/ruiyang/Data/coco/annotations/instances_train2017.json'
    trainset = Coco(root, annFile)