export HDF5_USE_FILE_LOCKING=FALSE

python -m detector.detect \
    --det_input /shared/rsaas/tgupta6/Data/context-regions/coco_proc/det_input_train.json \
    --out_dir /shared/rsaas/tgupta6/Data/context-regions/coco_proc/detections/train

# python -m detector.detect \
#     --det_input /shared/rsaas/tgupta6/Data/context-regions/coco_proc/det_input_val.json \
#     --out_dir /shared/rsaas/tgupta6/Data/context-regions/coco_proc/detections/val