EXP_NAME="bert_negs_lang_loss_1_neg_loss_1_wo_detach_no_cheat"
COCO_EXP="/shared/rsaas/tgupta6/Data/context-regions/coco_exp"
COCO_PROC="/data/tgupta6/context-regions/coco_proc"
python -m exp.pretrain_coco_noun_negs.run.cache_context_features \
    --exp_name $EXP_NAME \
    --features_hdf5 "${COCO_PROC}/detections/val/features.hdf5" \
    --encoder_path "${COCO_EXP}/${EXP_NAME}/models/best_object_encoder" \
    --context_features_hdf5 "${COCO_EXP}/${EXP_NAME}/context_features_val_best.hdf5"