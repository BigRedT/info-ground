import os
from copy import deepcopy
from urllib.parse import urlparse

from global_constants import coco_paths
from utils.constants import Constants


class CocoConstants(Constants):
    def __init__(
            self,
            coco_paths=coco_paths):
        super().__init__()
        self.coco_paths = deepcopy(coco_paths)
        self.image_subset_dir = {}
        for subset in ['train','val','test']:
            self.image_subset_dir[subset] = os.path.join(
                self.coco_paths['image_dir'],
                self.coco_paths['extracted']['images'][subset])

        self.caption_annos_json = {}
        for subset in ['train','val']:
            self.caption_annos_json[subset] = os.path.join(
                self.coco_paths['proc_dir'],
                self.coco_paths['extracted']['annos']['captions'][subset])


if __name__=='__main__':
    # quick test to verify file paths
    const = CocoConstants()
    print(const.to_json())
