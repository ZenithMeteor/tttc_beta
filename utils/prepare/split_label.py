import os
import sys

import cv2 as cv
import numpy as np
from tqdm import tqdm

sys.path.append(os.getcwd())
from utils.prepare.putils import orderConvex, shrink_poly

# DATA_FOLDER = "data/dataset/mlt/"
LIST_FOLDER = "data/dataset/17/"
DATA_FOLDER = "/home/msp/Downloads/zadays/dataset/MLST_training_img/"
LABEL_FOLDER = "/home/msp/Downloads/zadays/dataset/MLST_training_labels/"
OUTPUT = "data/dataset/mlt8/"
#OUTPUT = "data/dataset/mlt/"
# gggggt
MAX_LEN = 1200
MIN_LEN = 600
stride = 8

im_fns = os.listdir(os.path.join(LIST_FOLDER))
im_fns.sort()

if not os.path.exists(os.path.join(OUTPUT, "image")):
    os.makedirs(os.path.join(OUTPUT, "image"))
if not os.path.exists(os.path.join(OUTPUT, "label")):
    os.makedirs(os.path.join(OUTPUT, "label"))
# im_fns=im_fns[:10]
for im_fn in tqdm(im_fns):
    # try:
    _, fn = os.path.split(im_fn)
    bfn, ext = os.path.splitext(fn)
    if ext.lower() not in ['.jpg', '.png']:
        continue

    # gt_path = os.path.join(DATA_FOLDER, "label", 'gt_' + bfn + '.txt')
    # img_path = os.path.join(DATA_FOLDER, "image", im_fn)
    gt_path = os.path.join(LABEL_FOLDER, 'gt_' + bfn + '.txt')
    img_path = os.path.join(DATA_FOLDER, im_fn)

    img = cv.imread(img_path)
    img_size = img.shape
    im_size_min = np.min(img_size[0:2])
    im_size_max = np.max(img_size[0:2])

    im_scale = float(600) / float(im_size_min)
    if np.round(im_scale * im_size_max) > 1200:
        im_scale = float(1200) / float(im_size_max)
    new_h = int(img_size[0] * im_scale)
    new_w = int(img_size[1] * im_scale)

    new_h = new_h if new_h // 16 == 0 else (new_h // 16 + 1) * 16
    new_w = new_w if new_w // 16 == 0 else (new_w // 16 + 1) * 16

    re_im = cv.resize(img, (new_w, new_h), interpolation=cv.INTER_LINEAR)
    re_size = re_im.shape

    polys = []
    with open(gt_path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        splitted_line = line.strip().lower().split(',')

        if splitted_line[9]=='###':
            continue
        else:
            x1, y1, x2, y2, x3, y3, x4, y4 = map(float, splitted_line[:8])
            poly = np.array([x1, y1, x2, y2, x3, y3, x4, y4]).reshape([4, 2])
            poly[:, 0] = poly[:, 0] / img_size[1] * re_size[1]
            poly[:, 1] = poly[:, 1] / img_size[0] * re_size[0]
            poly = orderConvex(poly)
            polys.append(poly)

        # cv.polylines(re_im, [poly.astype(np.int32).reshape((-1, 1, 2))], True,color=(0, 255, 0), thickness=2)

    res_polys = []
    for poly in polys:
        """ delete polys with width less than 10 pixel """
        if np.linalg.norm(poly[0] - poly[1]) < 10 or np.linalg.norm(poly[3] - poly[0]) < 10:
            continue

        res = shrink_poly(poly, stride)
        # for p in res:
        #    cv.polylines(re_im, [p.astype(np.int32).reshape((-1, 1, 2))], True, color=(0, 255, 0), thickness=1)

        res = res.reshape([-1, 4, 2])
        for r in res:
            x_min = np.min(r[:, 0])
            y_min = np.min(r[:, 1])
            x_max = np.max(r[:, 0])
            y_max = np.max(r[:, 1])

            res_polys.append([x_min, y_min, x_max, y_max])
            # cv.rectangle(re_im, (x_min,y_min), (x_max, y_max), color=(0, 255, 0), thickness=1)

    cv.imwrite(os.path.join(OUTPUT, "image", fn), re_im)
    with open(os.path.join(OUTPUT, "label", bfn) + ".txt", "w") as f:
        for p in res_polys:
            line = ",".join(str(p[i]) for i in range(4))
            f.writelines(line + "\r\n")
            # for p in res_polys:
            #    cv.rectangle(re_im,(p[0],p[1]),(p[2],p[3]),color=(0,0,255),thickness=1)

            # cv.imshow("demo",re_im)
            # cv.waitKey(0)
    # except:
    #     print("Error processing {}".format(im_fn))
