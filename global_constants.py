import yaml

machine='ngc' # {'ngc','htc'}

if machine=='htc':
    coco_paths = yaml.load(
        open('yaml/coco.yml','r'),
        Loader=yaml.FullLoader)

    flickr_paths = yaml.load(
        open('yaml/flickr.yml','r'),
        Loader=yaml.FullLoader)

    misc_paths = yaml.load(
        open('yaml/misc.yml','r'),
        Loader=yaml.FullLoader)

elif machine=='ngc':
    coco_paths = yaml.load(
        open('yaml/coco_ngc.yml','r'),
        Loader=yaml.FullLoader)

    flickr_paths = yaml.load(
        open('yaml/flickr_ngc.yml','r'),
        Loader=yaml.FullLoader)

    misc_paths = yaml.load(
        open('yaml/misc_ngc.yml','r'),
        Loader=yaml.FullLoader)

else:
    assert(False),'Machine not implemented'