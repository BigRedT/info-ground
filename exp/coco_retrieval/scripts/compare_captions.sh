EXP_DIR="${PWD}/symlinks/rsaas/coco_exp/bert_negs_lang_loss_1_neg_loss_1_wo_detach"
KNN_JSON="${EXP_DIR}/knn_context_features_val_best.json"
python -m exp.coco_retrieval.compare_captions \
    --knn_json $KNN_JSON \
    --k 3


# EXP_DIR="${PWD}/symlinks/rsaas/coco_exp/uncontextualized_retrieval"
# KNN_JSON="${EXP_DIR}/knn_object_features_val.json"
# python -m exp.coco_retrieval.compare_captions \
#     --knn_json $KNN_JSON \
#     --k 3


# EXP_DIR="${PWD}/symlinks/rsaas/coco_exp/whole_image_mean_object_feat_retrieval"
# KNN_JSON="${EXP_DIR}/knn_object_features_val.json"
# python -m exp.coco_retrieval.compare_captions \
#     --knn_json $KNN_JSON \
#     --k 3


# EXP_DIR="${PWD}/symlinks/rsaas/coco_exp/whole_image_concat_mean_object_feat_retrieval"
# KNN_JSON="${EXP_DIR}/knn_object_features_val.json"
# python -m exp.coco_retrieval.compare_captions \
#     --knn_json $KNN_JSON \
#     --k 3