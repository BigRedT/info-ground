export HDF5_USE_FILE_LOCKING=FALSE

python -m exp.pretrain_coco_negs.run.train \
    --exp_name 'bert_negs_lang_loss_1_neg_noun_loss_1_neg_verb_loss_1' \
    --model_num -1 \
    --lr 1e-5 \
    --train_batch_size 50 \
    --neg_verb_loss_wt 1 \
    --neg_noun_loss_wt 1 \
    --self_sup_loss_wt 0 \
    --lang_sup_loss_wt 1