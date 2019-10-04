from detector.model import COCO_INSTANCE_CATEGORY_NAMES as DETECTOR_COCO_CLASSES

HICO_COCO_CLASSES = (
    'background',
    'person',
    'bicycle',
    'car',
    'motorcycle',
    'airplane',
    'bus',
    'train',
    'truck',
    'boat',
    'traffic light',
    'fire hydrant',
    'stop sign',
    'parking meter',
    'bench',
    'bird',
    'cat',
    'dog',
    'horse',
    'sheep',
    'cow',
    'elephant',
    'bear',
    'zebra',
    'giraffe',
    'backpack',
    'umbrella',
    'handbag',
    'tie',
    'suitcase',
    'frisbee',
    'skis',
    'snowboard',
    'sports ball',
    'kite',
    'baseball bat',
    'baseball glove',
    'skateboard',
    'surfboard',
    'tennis racket',
    'bottle',
    'wine glass',
    'cup',
    'fork',
    'knife',
    'spoon',
    'bowl',
    'banana',
    'apple',
    'sandwich',
    'orange',
    'broccoli',
    'carrot',
    'hot dog',
    'pizza',
    'donut',
    'cake',
    'chair',
    'couch',
    'potted plant',
    'bed',
    'dining table',
    'toilet',
    'tv',
    'laptop',
    'mouse',
    'remote',
    'keyboard',
    'cell phone',
    'microwave',
    'oven',
    'toaster',
    'sink',
    'refrigerator',
    'book',
    'clock',
    'vase',
    'scissors',
    'teddy bear',
    'hair drier',
    'toothbrush')

det_cls_id_to_coco_cls_id = [None]*len(DETECTOR_COCO_CLASSES)
last_match = -1
for coco_cls_id,coco_cls in enumerate(HICO_COCO_CLASSES):
    for det_cls_id in range(last_match+1,len(DETECTOR_COCO_CLASSES)):
        det_cls = DETECTOR_COCO_CLASSES[det_cls_id]
        if coco_cls in det_cls:
            det_cls_id_to_coco_cls_id[det_cls_id] = coco_cls_id
            last_match = det_cls_id
            break

# for det_cls_id, coco_cls_id in enumerate(det_cls_id_to_coco_cls_id):
#     det_cls = DETECTOR_COCO_CLASSES[det_cls_id]
#     if coco_cls_id is None:
#         coco_cls = "NONE"
#     else:
#         coco_cls = HICO_COCO_CLASSES[coco_cls_id]

#     print(det_cls,coco_cls)

