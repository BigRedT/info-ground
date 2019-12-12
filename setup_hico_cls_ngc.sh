NFS_DATA="/home/workspace/Data/context-regions/"
DATA="/dev/shm/context-regions"
mkdir -p $DATA

# Copy features
HICO_PROC="${DATA}/hico_proc"
mkdir $HICO_PROC
cp -r "${NFS_DATA}/hico_proc/detections" $HICO_PROC