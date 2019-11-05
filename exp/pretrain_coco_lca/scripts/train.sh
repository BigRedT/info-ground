export HDF5_USE_FILE_LOCKING=FALSE

python -m exp.pretrain_coco_lca.run.train \
    --exp_name 'lca_lang_sup_gumbel_0.5_finetune_lang_margin' \
    --model_num 96000 \
    --lr 1e-5 \
    --train_batch_size 50