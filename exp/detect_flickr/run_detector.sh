export HDF5_USE_FILE_LOCKING=FALSE
SUBSET='test'
python -m detector.detect \
    --det_input /shared/rsaas/tgupta6/Data/context-regions/flickr30k_proc/det_input_$SUBSET.json \
    --out_dir /shared/rsaas/tgupta6/Data/context-regions/flickr30k_proc/detections/$SUBSET