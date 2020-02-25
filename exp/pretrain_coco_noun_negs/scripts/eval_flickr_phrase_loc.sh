export HDF5_USE_FILE_LOCKING=FALSE

LAYERS=2
python -m exp.pretrain_coco_noun_negs.run.eval_flickr_phrase_loc \
    --exp_name "loss_wts_neg_noun_0_self_sup_1_lang_sup_1_no_context_vgdet_nonlinear_infonce_${LAYERS}_layer_adj_batch_50_random_lang" \
    --model_num -100 \
    --no_context \
    --cap_info_nce_layers $LAYERS \
    --random_lang \
    --subset 'test'