export HDF5_USE_FILE_LOCKING=FALSE

python -m exp.pretrain_coco_negs.run.train \
    --exp_name 'bert_negs_lang_loss_1_neg_loss_1_same_att_V_o_wo_detach' \
    --model_num -1 \
    --lr 1e-5 \
    --train_batch_size 50