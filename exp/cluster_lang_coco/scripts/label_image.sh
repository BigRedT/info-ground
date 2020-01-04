CLUSTER_DIR="/shared/rsaas/tgupta6/Data/context-regions/coco_exp/bert_noun_clusters_500"
python -m exp.cluster_lang_coco.label_images \
    --feat_info_json "${CLUSTER_DIR}/feat_info.json" \
    --active_words_json "${CLUSTER_DIR}/active_nouns.json" \
    --image_labels_json "${CLUSTER_DIR}/image_labels.json" \
    --labels_json "${CLUSTER_DIR}/labels.json"