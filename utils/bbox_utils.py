import numpy as np
import skimage.draw as skdraw

def vis_bbox(bbox,img,color=(255,0,0),modify=False,alpha=0.2):
    im_h,im_w = img.shape[0:2]
    x1,y1,x2,y2 = bbox
    x1 = max(0,min(x1,im_w-1))
    x2 = max(x1,min(x2,im_w-1))
    y1 = max(0,min(y1,im_h-1))
    y2 = max(y1,min(y2,im_h-1))
    r = [y1,y1,y2,y2]
    c = [x1,x2,x2,x1]

    if modify:
        img_ = img
    else:
        img_ = np.copy(img)

    if len(img.shape)==2:
        color = (color[0],)

    rr,cc = skdraw.polygon(r,c,img.shape[:2])
    skdraw.set_color(img_,(rr,cc),color,alpha=alpha)

    rr,cc = skdraw.polygon_perimeter(r,c,img.shape[:2])
    
    if len(img.shape)==3:
        for k in range(3):
            img_[rr,cc,k] = color[k]
    elif len(img.shape)==2:
        img_[rr,cc]=color[0]

    return img_


def create_att(bbox,prev_att,att_value):
    im_h,im_w = prev_att.shape[0:2]
    x1,y1,x2,y2 = bbox
    x1 = int(max(0,min(x1,im_w-1)))
    x2 = int(max(x1,min(x2,im_w-1)))
    y1 = int(max(0,min(y1,im_h-1)))
    y2 = int(max(y1,min(y2,im_h-1)))
    r = [y1,y1,y2,y2]
    c = [x1,x2,x2,x1]
    att = 0*prev_att
    att[y1:y2,x1:x2] = att_value
    return np.maximum(prev_att,att)