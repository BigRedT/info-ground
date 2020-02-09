export HDF5_USE_FILE_LOCKING=FALSE


python -m exp.pretrain_coco_noun_negs_sidetune.run.train \
    --exp_name 'loss_wts_neg_noun_1_self_sup_1_lang_sup_1_side_norm_1e-2_no_context_self_sup_feat_sidetune_alpha_init_0.8_mad_0.2' \
    --model_num -1 \
    --lr 1e-5 \
    --train_batch_size 50 \
    --neg_noun_loss_wt 1 \
    --self_sup_loss_wt 1 \
    --lang_sup_loss_wt 1 \
    --no_context \
    --self_sup_feat
    #--drop_prob 0.5

# python -m exp.pretrain_coco_noun_negs.run.train \
#     --exp_name 'loss_wts_neg_noun_1_self_sup_1_lang_sup_1' \
#     --model_num -1 \
#     --lr 1e-5 \
#     --train_batch_size 50 \
#     --neg_noun_loss_wt 1 \
#     --self_sup_loss_wt 1 \
#     --lang_sup_loss_wt 1

# python -m exp.pretrain_coco_noun_negs.run.train \
#     --exp_name 'loss_wts_neg_noun_1_self_sup_0_lang_sup_1' \
#     --model_num -1 \
#     --lr 1e-5 \
#     --train_batch_size 50 \
#     --neg_noun_loss_wt 1 \
#     --self_sup_loss_wt 0 \
#     --lang_sup_loss_wt 1