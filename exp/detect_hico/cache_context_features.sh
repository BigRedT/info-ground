EXP_NAME="lca_lang_sup_gumbel_0.5_finetune"
HICO_PROC="/shared/rsaas/tgupta6/Data/no_frills/hico_proc"
HICO_FEAT="/shared/rsaas/tgupta6/Data/no_frills/hico_feat"
COCO_EXP="/shared/rsaas/tgupta6/Data/context-regions/coco_exp"
python -m exp.pretrain_coco_lca.run.cache_context_features \
    --exp_name $EXP_NAME \
    --features_hdf5 "${HICO_FEAT}/features.hdf5" \
    --encoder_path "${COCO_EXP}/${EXP_NAME}/models/best_object_encoder" \
    --context_features_hdf5 "${HICO_FEAT}/${EXP_NAME}_context_features.hdf5"