import os
from copy import deepcopy
from urllib.parse import urlparse

from global_constants import flickr_paths
from utils.constants import Constants


class FlickrConstants(Constants):
    flickr_paths = deepcopy(flickr_paths)
    
    def __init__(self,flickr_paths=flickr_paths):
        super().__init__()
        self.subset_ids = {}
        self.box_json = {}
        self.sent_json = {}
        for subset in ['train','val','test']:
            self.subset_ids[subset] = os.path.join(
                self.flickr_paths['downloads_dir'],
                self.flickr_paths['subsets'][subset])

            self.box_json[subset] = os.path.join(
                self.flickr_paths['proc_dir'],
                f'bounding_boxes_{subset}.json')
            
            self.sent_json[subset] = os.path.join(
                self.flickr_paths['proc_dir'],
                f'sentences_{subset}.json')


if __name__=='__main__':
    const = FlickrConstants()
    import pdb; pdb.set_trace()