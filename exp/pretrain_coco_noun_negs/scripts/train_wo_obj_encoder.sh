export HDF5_USE_FILE_LOCKING=FALSE

python -m exp.pretrain_coco_noun_negs.run.train_wo_obj_encoder \
    --exp_name 'bert_negs_wo_obj_encoder' \
    --model_num -1 \
    --lr 1e-5 \
    --train_batch_size 50 \
    --neg_verb_loss_wt 0 \
    --neg_noun_loss_wt 0 \
    --self_sup_loss_wt 0 \
    --lang_sup_loss_wt 1