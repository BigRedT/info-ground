export HDF5_USE_FILE_LOCKING=FALSE

EXP_NAME="self_factor_lang_sup_train_batch_size_50"
HICO_FEAT="/home/tgupta6/Code/no_frills_hoi_det_context/data_symlinks/hico_feat"
python -m exp.detect_hico.concat_object_context \
    --obj_feat_hdf5 "${HICO_FEAT}/features.hdf5" \
    --context_feat_hdf5 "${HICO_FEAT}/${EXP_NAME}_context_features.hdf5" \
    --concat_feat_hdf5 "${HICO_FEAT}/${EXP_NAME}_concat_features.hdf5"