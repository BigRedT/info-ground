DOWNLOAD_DIR="/shared/rsaas/tgupta6/Data/info-ground/flickr30k_downloads"
git clone git@github.com:BryanPlummer/flickr30k_entities.git $DOWNLOAD_DIR

annotations_zip="${DOWNLOAD_DIR}/annotations.zip"
unzip $annotations_zip -d $DOWNLOAD_DIR