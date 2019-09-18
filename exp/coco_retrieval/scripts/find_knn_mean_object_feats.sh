EXP_DIR="/home/tgupta6/Code/context-regions/symlinks/rsaas/coco_exp/whole_image_concat_mean_object_feat_retrieval"
PROC_DIR="/home/tgupta6/Code/context-regions/symlinks/rsaas/coco_proc"
SUBSET="val"

mkdir $EXP_DIR

python -m exp.coco_retrieval.find_knn \
    --object_index_json "${PROC_DIR}/object_index_${SUBSET}.json" \
    --boxes_hdf5 "${PROC_DIR}/detections/${SUBSET}/boxes.hdf5" \
    --features_hdf5 "${PROC_DIR}/detections/${SUBSET}/concat_mean_features.hdf5" \
    --search_objects_json "${EXP_DIR}/search_objects_${SUBSET}.json" \
    --search_features_hdf5 "${EXP_DIR}/search_features_${SUBSET}.hdf5" \
    --num_queries 20 \
    --num_nbrs 3 \
    --anno_subset $SUBSET \
    --knn_json "${EXP_DIR}/knn_object_features_${SUBSET}.json" \
    --vis_dir "${EXP_DIR}/vis/knn_object_features/${SUBSET}" #\
    #--load_knn_json
