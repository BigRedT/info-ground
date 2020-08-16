export HDF5_USE_FILE_LOCKING=FALSE

LAYERS=2
python -m exp.ground.run.eval_flickr_phrase_loc \
    --exp_name "model_trained_on_coco" \
    --model_num -100 \
    --no_context \
    --cap_info_nce_layers $LAYERS \
    --subset 'test'