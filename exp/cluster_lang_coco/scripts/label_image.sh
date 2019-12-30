CLUSTER_DIR="/home/workspace/Data/context-regions/coco_exp/bert_noun_clusters_2000"
python -m exp.cluster_lang_coco.label_images \
    --feat_info_json "${CLUSTER_DIR}/feat_info.json" \
    --active_words_json "${CLUSTER_DIR}/active_nouns.json" \
    --image_labels_json "${CLUSTER_DIR}/image_labels.json" \
    --labels_json "${CLUSTER_DIR}/labels.json"