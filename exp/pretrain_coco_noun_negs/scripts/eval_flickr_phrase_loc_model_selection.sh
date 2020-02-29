export HDF5_USE_FILE_LOCKING=FALSE

LAYERS=2
python -m exp.pretrain_coco_noun_negs.run.eval_flickr_phrase_loc_model_selection \
    --exp_name "loss_wts_neg_noun_1_self_sup_1_lang_sup_1_no_context_vgdet_nonlinear_infonce_${LAYERS}_layer_adj_batch_50_flickr" \
    --no_context \
    --cap_info_nce_layers $LAYERS \
    --subset val