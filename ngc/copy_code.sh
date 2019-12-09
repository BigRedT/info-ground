CODE_DIR='/Users/tanmay/Code/ngc/context-regions'
WS_DIR='/Users/tanmay/Data/ngc_workspaces/ws-tanmayg/Code'
#rsync -av --exclude=".git/" --modify-window=10 $CODE_DIR $WS_DIR

pushd '/Users/tanmay/Code/ngc/'
zip -rq /tmp/code.zip context-regions
rsync -rv /tmp/code.zip $WS_DIR
rm /tmp/code.zip
popd

rsync $CODE_DIR/ngc/unzip_context_regions.sh $WS_DIR
