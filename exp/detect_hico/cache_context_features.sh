EXP_NAME="bert_negs_lang_loss_1_neg_noun_loss_1_neg_verb_loss_1"
HICO_PROC="/shared/rsaas/tgupta6/Data/no_frills/hico_proc"
HICO_FEAT="/shared/rsaas/tgupta6/Data/no_frills/hico_feat"
COCO_EXP="/shared/rsaas/tgupta6/Data/context-regions/coco_exp"
python -m exp.pretrain_coco_negs.run.cache_context_features \
    --exp_name $EXP_NAME \
    --features_hdf5 "${HICO_FEAT}/features.hdf5" \
    --encoder_path "${COCO_EXP}/${EXP_NAME}/models/best_object_encoder" \
    --context_features_hdf5 "${HICO_FEAT}/${EXP_NAME}_context_features.hdf5"