EXP_NAME=$1
DATASET=$2
export HDF5_USE_FILE_LOCKING=FALSE

python -m exp.ground.run.train \
    --exp_name $EXP_NAME \
    --dataset $DATASET \
    --model_num -1 \
    --neg_noun_loss_wt 1 \
    --self_sup_loss_wt 0 \
    --lang_sup_loss_wt 1 \
    --no_context \
    --cap_info_nce_layers 2