export HDF5_USE_FILE_LOCKING=FALSE
SUBSET=$1
python -m exp.cmc_coco.extract_features \
    --subset $SUBSET \
    --num_workers 20 \
    --batch_size 20