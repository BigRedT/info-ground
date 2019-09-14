EXP_DIR='/home/tgupta6/Code/context-regions/symlinks/rsaas/coco_exp/self_sup_train_batch_size_200'
PROC_DIR='/home/tgupta6/Code/context-regions/symlinks/rsaas/coco_proc'
python -m exp.coco_retrieval.find_knn \
    --object_index_json "${PROC_DIR}/object_index_val.json" \
    --features_hdf5 "${EXP_DIR}/context_features_val_best.hdf5" \
    --search_objects_json "${EXP_DIR}/search_objects_val_best.json" \
    --search_features_hdf5 "${EXP_DIR}/search_features_val_best.hdf5" \
    --num_queries 10 \
    --num_nbrs 3 \
    --anno_subset 'val' \
    --knn_json "${EXP_DIR}/knn_context_features_val_best.json" \
    --vis_dir "${EXP_DIR}/vis/knn"