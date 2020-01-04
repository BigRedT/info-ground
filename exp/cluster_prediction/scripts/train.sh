export HDF5_USE_FILE_LOCKING=FALSE

python -m exp.cluster_prediction.run.train \
    --exp_name 'cluster_pred_att_adam_det_distil_bert_noun_clusters_2000_lr_1e-5_20_boxes_wsddn_pos_neg' \
    --model_num -1 \
    --lr 1e-5 \
    --train_batch_size 5

#cluster_pred_att_adam_det_no_distil_bert_noun_clusters_500_lr_1e-5_20_boxes_wsddn_pos_neg
#cluster_pred_att_adam_det_no_distil_bert_noun_clusters_2000_lr_1e-5_20_boxes_wsddn_pos_neg
#cluster_pred_att_adam_det_distil_bert_noun_clusters_2000_lr_1e-5_20_boxes_wsddn_pos_neg