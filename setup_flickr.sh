NFS_DATA="/shared/rsaas/tgupta6/Data/context-regions"
DATA="/data/tgupta6/context-regions"
mkdir -p $DATA

# Untar images to local data dir
FLICKR_IMAGES="${DATA}/flickr30k_images"
mkdir $FLICKR_IMAGES
tar -xvf "${NFS_DATA}/flickr30k_downloads/flickr30k-images.tar" -C $FLICKR_IMAGES
mv $FLICKR_IMAGES/flickr30k-images/* $FLICKR_IMAGES
rm -rf $FLICKR_IMAGES/flickr30k-images

# Copy detections to local data dir
FLICKR_PROC="${DATA}/flickr30k_proc"
mkdir $FLICKR_PROC
cp -r "${NFS_DATA}/flickr30k_proc/bottomup_flickr_detections" $FLICKR_PROC

# Copy cache features
cp -r "${NFS_DATA}/flickr30k_proc/bert_noun_negatives_train.h5py" $FLICKR_PROC
cp -r "${NFS_DATA}/flickr30k_proc/bert_noun_negatives_val.h5py" $FLICKR_PROC