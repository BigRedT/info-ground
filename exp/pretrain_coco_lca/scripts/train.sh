export HDF5_USE_FILE_LOCKING=FALSE

python -m exp.pretrain_coco_lca.run.train \
    --exp_name 'lca_lang_sup_att_loss_100' \
    --model_num -1 \
    --lr 1e-5 \
    --train_batch_size 50