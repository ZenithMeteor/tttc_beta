# encoding:utf-8
import os
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np

from utils.dataset.data_util import GeneratorEnqueuer
from utils import helpers

DATA_FOLDER = "data/dataset/mlt8/"


def get_training_data():
    img_files = []
    exts = ['jpg', 'png', 'jpeg', 'JPG']
    for parent, dirnames, filenames in os.walk(os.path.join(DATA_FOLDER, "image")):
        for filename in filenames:
            for ext in exts:
                if filename.endswith(ext):
                    img_files.append(os.path.join(parent, filename))
                    break
    print('Find {} images'.format(len(img_files)))
    return img_files


def load_annoataion(p):
    bbox = []
    with open(p, "r") as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip().split(",")
        x_min, y_min, x_max, y_max = map(int, line)
        bbox.append([x_min, y_min, x_max, y_max, 1])
    return bbox


def dilate_se(lx,ly,rx,ry,_):
    # coefficient of dilatation
    exx=3
    exy=1

    w = rx-lx
    h = ry-ly
    cx = w/2 + lx
    cy = h/2 + ly
    lx = int(cx - w*exx)
    rx = int(cx + w*exx)
    ly = int(cy - h*exy)
    ry = int(cy + h*exy)
    return lx,ly,rx,ry


def generate_Se_gt(im_info, bbox, dilate=False):
    gt = np.zeros((im_info[0][0], im_info[0][1], 1), np.uint8)
    for p in bbox:
        if dilate:
            lx,ly,rx,ry=dilate_se(*p)
            p[0] = lx if lx >= 0 else 0
            p[2] = rx if rx <= im_info[0][1] else im_info[0][1]
            p[1] = ly if ly >= 0 else 0
            p[3] = ry if ry <= im_info[0][0] else im_info[0][0]

        cv2.rectangle(gt, (p[0], p[1]), (p[2], p[3]), color=(100, 100, 100), thickness=2)#-1
    return gt


def generator(label_values, dilate, vis=False):
    image_list = np.array(get_training_data())
    print('{} training images in {}'.format(image_list.shape[0], DATA_FOLDER))
    index = np.arange(0, image_list.shape[0])
    while True:
        np.random.shuffle(index)
        for i in index:
            try:
                im_fn = image_list[i]
                im = cv2.imread(im_fn)
                h, w, c = im.shape
                im_info = np.array([h, w, c]).reshape([1, 3])

                _, fn = os.path.split(im_fn)
                fn, _ = os.path.splitext(fn)
                txt_fn = os.path.join(DATA_FOLDER, "label", fn + '.txt')
                if not os.path.exists(txt_fn):
                    print("Ground truth for image {} not exist!".format(im_fn))
                    continue
                bbox = load_annoataion(txt_fn)
                if len(bbox) == 0:
                    print("Ground truth for image {} empty!".format(im_fn))
                    continue
                ### get label
                gt = generate_Se_gt(im_info, bbox, dilate)
                output_label = np.float32(helpers.one_hot_it(label=gt, label_values=label_values))
                #im = np.float32(im) / 255.0

                if vis:
                    for p in bbox:
                        cv2.rectangle(im, (p[0], p[1]), (p[2], p[3]), color=(0, 0, 255), thickness=1)
                    fig, axs = plt.subplots(1, 1, figsize=(30, 30))
                    axs.imshow(im[:, :, ::-1])
                    axs.set_xticks([])
                    axs.set_yticks([])
                    plt.tight_layout()
                    plt.show()
                    plt.close()
                    cv2.imwrite('./tt/'+fn+'.png',gt)
                yield [im], bbox, im_info ,[output_label]

            except Exception as e:
                print(e)
                continue


def get_batch(num_workers, dilate, **kwargs):
    try:
        csv_path = './data/dataset/'
        # Get the names of the classes so we can record the evaluation results
        class_names_list, label_values = helpers.get_label_info(os.path.join(csv_path, "class_dict.csv"))
        class_names_string = ""
        for class_name in class_names_list:
            if not class_name == class_names_list[-1]:
                class_names_string = class_names_string + class_name + ", "
            else:
                class_names_string = class_names_string + class_name

        num_classes = len(label_values)

        enqueuer = GeneratorEnqueuer(generator(label_values, dilate, **kwargs), use_multiprocessing=True)
        enqueuer.start(max_queue_size=24, workers=num_workers)
        generator_output = None
        while True:
            while enqueuer.is_running():
                if not enqueuer.queue.empty():
                    generator_output = enqueuer.queue.get()
                    break
                else:
                    time.sleep(0.01)
            yield generator_output
            generator_output = None
    finally:
        if enqueuer is not None:
            enqueuer.stop()


if __name__ == '__main__':
    gen = get_batch(num_workers=2, dilate=True, vis=True)
    while True:
        image, bbox, im_info, _ = next(gen)
        print('done')
