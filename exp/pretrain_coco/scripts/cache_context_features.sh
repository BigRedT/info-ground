python -m exp.pretrain_coco.run.cache_context_features \
    --exp_name 'self_sup_train_batch_size_200' \
    --features_hdf5 '/data/tgupta6/context-regions/coco_proc/detections/val/features.hdf5' \
    --encoder_path '/shared/rsaas/tgupta6/Data/context-regions/coco_exp/self_sup_train_batch_size_200/models/best_object_encoder' \
    --context_features_hdf5 'context_features_val_best.hdf5'