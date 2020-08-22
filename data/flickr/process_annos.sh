echo "Processing train set"
python -m data.flickr.write_annos_to_json --subset train
echo "Processing val set"
python -m data.flickr.write_annos_to_json --subset val
echo "Processing test set"
python -m data.flickr.write_annos_to_json --subset test