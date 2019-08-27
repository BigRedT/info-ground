import os
from zipfile import ZipFile
from urllib.parse import urlparse

from global_constants import coco_paths


subsets = ['test']
for subset in subsets:
    url = coco_paths['urls']['images'][subset]
    filename = os.path.join(
        coco_paths['downloads_dir'],
        os.path.basename(urlparse(url).path))
    print('Extracting from:',filename,'to:',coco_paths['image_dir'])
    with ZipFile(filename,'r') as z:
        z.extractall(coco_paths['image_dir'])
