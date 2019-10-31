export HDF5_USE_FILE_LOCKING=FALSE

EXP_NAME="lca_lang_sup_gumbel_0.5_finetune"
HICO_FEAT="/shared/rsaas/tgupta6/Data/no_frills/hico_feat"
python -m exp.detect_hico.concat_object_context \
    --obj_feat_hdf5 "${HICO_FEAT}/features.hdf5" \
    --context_feat_hdf5 "${HICO_FEAT}/${EXP_NAME}_context_features.hdf5" \
    --concat_feat_hdf5 "${HICO_FEAT}/${EXP_NAME}_concat_features.hdf5"