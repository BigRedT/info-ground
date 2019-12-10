import os
import wget
import tarfile

from global_constants import hico_paths
import utils.io as io


print('Creating directories ...')
for dir_type in ['downloads_dir','image_dir','proc_dir','local_proc_dir']:
    dirname = hico_paths[dir_type]
    print('- ',dirname)
    io.mkdir_if_not_exists(dirname,recursive=True)

print('Extracting tar.gz ...')
z = tarfile.open(hico_paths['hico_tar_gz'])

print('Extracting images ...')
image_dir = hico_paths['image_dir']
downloads_dir = hico_paths['downloads_dir']
z.extractall(image_dir)

print('Moving images to image_dir ...')
os.system(f"mv {image_dir}/hico_20150920/images/* {image_dir}")

print('Moving annotations to downloads_dir ...')
os.system(f"mv {image_dir}/hico_20150920/anno.mat {downloads_dir}")
os.system(f"mv {image_dir}/hico_20150920/README {downloads_dir}")

print('Deleting empty directory ...')
os.system(f"rm -rf {image_dir}/hico_20150920")