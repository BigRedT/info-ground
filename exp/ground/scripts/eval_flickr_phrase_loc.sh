EXP_NAME=$1
DATASET=$2
export HDF5_USE_FILE_LOCKING=FALSE

LAYERS=2
python -m exp.ground.run.eval_flickr_phrase_loc \
    --exp_name $EXP_NAME \
    --dataset $DATASET \
    --model_num -100 \
    --no_context \
    --cap_info_nce_layers $LAYERS \
    --subset 'test'