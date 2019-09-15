PROC_DIR="/home/tgupta6/Code/context-regions/symlinks/rsaas/coco_proc"
SUBSET="val"
python -m exp.coco_retrieval.create_object_index \
    --labels_hdf5 "${PROC_DIR}/detections/${SUBSET}/labels.hdf5" \
    --scores_hdf5 "${PROC_DIR}/detections/${SUBSET}/scores.hdf5" \
    --object_index_json "${PROC_DIR}/object_index_${SUBSET}.json"