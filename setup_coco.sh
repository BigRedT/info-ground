NFS_DATA="/shared/rsaas/tgupta6/Data/context-regions/"
DATA="/data/tgupta6/context-regions"
mkdir -p $DATA

# Copy images to local data dir
COCO_IMAGES="${DATA}/coco_images"
mkdir $COCO_IMAGES
unzip "${NFS_DATA}/coco_downloads/train2014.zip" -d $COCO_IMAGES
unzip "${NFS_DATA}/coco_downloads/val2014.zip" -d $COCO_IMAGES

# Copy detections to local data dir
COCO_PROC="${DATA}/coco_proc"
mkdir $COCO_PROC
cp -r "${NFS_DATA}/coco_proc/bottomup_coco_detections" $COCO_PROC

# Copy cache features to local data dir
cp -r "${NFS_DATA}/coco_proc/bert_noun_negatives_train2014.h5py" $COCO_PROC
cp -r "${NFS_DATA}/coco_proc/bert_noun_negatives_val2014.h5py" $COCO_PROC