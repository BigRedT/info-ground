NFS_DATA="/shared/rsaas/tgupta6/Data/context-regions/"
DATA="/data/tgupta6/context-regions"
mkdir -p $DATA

# Copy images
COCO_IMAGES="${DATA}/coco_images"
mkdir $COCO_IMAGES
#unzip "${NFS_DATA}/coco_downloads/val2014.zip" -d $COCO_IMAGES

# Copy features
COCO_PROC="${DATA}/coco_proc"
mkdir $COCO_PROC
cp -r "${NFS_DATA}/coco_proc/detections" $COCO_PROC



