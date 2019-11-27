SUBSET=$1
export HDF5_USE_FILE_LOCKING=FALSE
python -m exp.gen_noun_negatives.cache_neg_features --subset $SUBSET
#python -m exp.gen_negatives.cache_neg_features --subset val