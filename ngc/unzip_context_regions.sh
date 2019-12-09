# CODE_DIR='/home/workspace/Code'
# rm -rf $CODE_DIR/context-regions
# unzip $CODE_DIR/code.zip -d $CODE_DIR
# rm $CODE_DIR/code.zip

CODE_DIR='/home/workspace/Code'
unzip -q $CODE_DIR/code.zip -d /tmp/
rsync -av --exclude=".git/" /tmp/context-regions $CODE_DIR
rm $CODE_DIR/code.zip
rm -rf /tmp/context-regions