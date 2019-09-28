export HDF5_USE_FILE_LOCKING=FALSE

python -m detector.detect \
    --det_input /home/tgupta6/Code/no_frills_hoi_det_context/data_symlinks/hico_exp/detect_coco_objects_in_hico/faster_rcnn_im_in_out.json \
    --out_dir /home/tgupta6/Code/no_frills_hoi_det_context/data_symlinks/hico_processed/faster_rcnn_boxes \
    --dataset 'hico'