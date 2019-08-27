import os
from zipfile import ZipFile
from urllib.parse import urlparse

from global_constants import coco_paths


subsets = ['train_val']
for subset in subsets:
    url = coco_paths['urls']['annos'][subset]
    filename = os.path.join(
        coco_paths['downloads_dir'],
        os.path.basename(urlparse(url).path))
    print('Extracting from:',filename,'to:',coco_paths['proc_dir'])
    with ZipFile(filename,'r') as z:
        z.extractall(coco_paths['proc_dir'])
