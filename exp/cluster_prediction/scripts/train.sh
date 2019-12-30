export HDF5_USE_FILE_LOCKING=FALSE

python -m exp.cluster_prediction.run.train \
    --exp_name 'cluster_pred_last_layer_bert_noun_clusters_2000_lr_1e-3' \
    --model_num -1 \
    --lr 1e-3 \
    --train_batch_size 5