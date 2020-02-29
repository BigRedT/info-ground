export HDF5_USE_FILE_LOCKING=FALSE

LAYERS=2
python -m exp.pretrain_coco_noun_negs.run.train_flickr \
    --exp_name "loss_wts_neg_noun_1_self_sup_1_lang_sup_1_no_context_vgdet_nonlinear_infonce_${LAYERS}_layer_adj_batch_50_flickr" \
    --model_num -1 \
    --lr 1e-5 \
    --train_batch_size 50 \
    --neg_noun_loss_wt 1 \
    --self_sup_loss_wt 1 \
    --lang_sup_loss_wt 1 \
    --no_context \
    --cap_info_nce_layers $LAYERS

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