EXP_NAME=$1
TRAIN_DATASET=$2
python -m exp.ground.run.vis_att \
    --exp_name $EXP_NAME \
    --train_dataset $TRAIN_DATASET \
    --vis_dataset 'flickr' \
    --model_num -100 \
    --no_context