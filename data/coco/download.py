import wget

from global_constants import coco_paths
import utils.io as io


print('Creating directories ...')
for dir_type in ['downloads_dir','image_dir','proc_dir']:
    dirname = coco_paths[dir_type]
    print('- ',dirname)
    io.mkdir_if_not_exists(dirname,recursive=True)


print('Downloading images ...')
subsets = ['train','val']
for subset in subsets:
    wget.download(
        coco_paths['urls']['images'][subset],
        out=coco_paths['downloads_dir'])


print('Downloading annotations ...')
for subset in ['train_val']:
    wget.download(
        coco_paths['urls']['annos'][subset],
        out=coco_paths['downloads_dir'])