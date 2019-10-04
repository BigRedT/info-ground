EXP_NAME="self_factor_lang_sup_train_batch_size_50"
HICO_PROC="/home/tgupta6/Code/no_frills_hoi_det_context/data_symlinks/hico_processed"
HICO_FEAT="/home/tgupta6/Code/no_frills_hoi_det_context/data_symlinks/hico_feat"
COCO_EXP="/shared/rsaas/tgupta6/Data/context-regions/coco_exp"
COCO_PROC="/data/tgupta6/context-regions/coco_proc"
python -m exp.pretrain_coco.run.cache_context_features \
    --exp_name $EXP_NAME \
    --features_hdf5 "${HICO_FEAT}/features.hdf5" \
    --encoder_path "${COCO_EXP}/${EXP_NAME}/models/best_object_encoder" \
    --context_features_hdf5 "${HICO_FEAT}/${EXP_NAME}_context_features.hdf5"