export HDF5_USE_FILE_LOCKING=FALSE

EXP_NAME="bert_negs_lang_loss_1_neg_noun_loss_1_neg_verb_loss_1"
HICO_FEAT="/shared/rsaas/tgupta6/Data/no_frills/hico_feat"
python -m exp.detect_hico.concat_object_context \
    --obj_feat_hdf5 "${HICO_FEAT}/features.hdf5" \
    --context_feat_hdf5 "${HICO_FEAT}/${EXP_NAME}_context_features.hdf5" \
    --concat_feat_hdf5 "${HICO_FEAT}/${EXP_NAME}_concat_features.hdf5"