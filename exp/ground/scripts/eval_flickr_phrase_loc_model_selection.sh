export HDF5_USE_FILE_LOCKING=FALSE

python -m exp.ground.run.eval_flickr_phrase_loc_model_selection \
    --exp_name "model_trained_on_coco" \
    --no_context \
    --cap_info_nce_layers 2 \
    --subset val