PROC_DIR="/home/tgupta6/Code/context-regions/symlinks/rsaas/coco_proc"
SUBSET="val"

python -m exp.coco_retrieval.compute_mean_object_features \
    --feat_h5py "${PROC_DIR}/detections/${SUBSET}/features.hdf5" \
    --out_feat_h5py "${PROC_DIR}/detections/${SUBSET}/concat_mean_features.hdf5" \
    --concat_with_object