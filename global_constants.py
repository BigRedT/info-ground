import yaml


coco_paths = yaml.load(
    open('yaml/coco.yml','r'),
    Loader=yaml.FullLoader)

flickr_paths = yaml.load(
    open('yaml/flickr.yml','r'),
    Loader=yaml.FullLoader)

misc_paths = yaml.load(
    open('yaml/misc.yml','r'),
    Loader=yaml.FullLoader)