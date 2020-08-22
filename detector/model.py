from .torchvision_detection.faster_rcnn import fasterrcnn_resnet50_fpn
from .torchvision_detection.faster_rcnn_extractor import fasterrcnn_resnet50_fpn \
    as fasterrcnn_resnet50_fpn_extractor


COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

def create_detector(dataset='coco',extractor=False,num_classes=91):
    if dataset in ['coco','default']:
        print('COCO/Default configuration for detector')
        if extractor==True:
            model = fasterrcnn_resnet50_fpn_extractor(
                pretrained=True,
                num_classes=num_classes)
        else:
            model = fasterrcnn_resnet50_fpn(pretrained=True)
    else:
        assert(False),'Dataset not implemented'

    return model
    

if __name__=='__main__':
    import torch
    imgs = [
        torch.rand([3,224,224]).cuda(),
        torch.rand([3,300,224]).cuda()
        ]
    props = [
        torch.FloatTensor([[10,20,60,80]]).cuda(),
        torch.FloatTensor([[10,20,60,80],[10,20,60,80]]).cuda()
        ]
    model = create_detector(extractor=True,num_classes=100).cuda()
    logits, region_logits, feats = model(imgs, props)
    import pdb; pdb.set_trace()
