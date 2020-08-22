download_dir="/shared/rsaas/tgupta6/Data/info-ground/flickr30k_downloads"
git clone git@github.com:BryanPlummer/flickr30k_entities.git $download_dir

annotations_zip="${download_dir}/annotations.zip"
unzip $annotations_zip -d $download_dir