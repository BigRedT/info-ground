SUBSET=$1
export HDF5_USE_FILE_LOCKING=FALSE
python -m exp.gen_noun_negatives.cache_neg_features_flickr --subset $SUBSET