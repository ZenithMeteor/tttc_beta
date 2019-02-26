import os
import numpy as np
import math
import cv2
import shutil

def compare(boxes, label, img, im_name):
    """
    # Arguments
        boxes: generated boxes
        label: predict se_img
        img: origin image
        im_name: origin image name
    # Returns
        None
        directly draw compared image
    """

    newboxes = []
    for i, box in enumerate(boxes):
        boxdone = False
        min_x = min(box[0], box[2])
        min_x = int(min(box[0], box[2]))
        max_x = int(max(box[4], box[6]))
        min_y = int(min(box[1], box[5]))
        max_y = int(max(box[3], box[7]))
        for x in range(min_x, max_x):
            for y in range(min_y, max_y):
                if label[y][x][0] == 100:
                    newboxes.append(box)
                    boxdone = True
                    break
            if boxdone == True:
                break

    newboxes = np.array(newboxes, dtype=np.int)
    for i, box in enumerate(newboxes):
        cv2.polylines(img, [newboxes[:8].astype(np.int32).reshape((-1, 1, 2))], True, color=(0, 255, 0),
                      thickness=2)
    cv2.imwrite(os.path.join('./data/test/', im_name + '.png'), img)

    # return newboxes
if __name__ == '__main__':
    path = os.getcwd() + '/data/test'
    demo_path = os.getcwd() + '/data/test/demo'
    files = os.listdir(demo_path)
    files.sort()
    for file in files:
        img = cv2.imread(os.path.join(demo_path ,file))
        _, basename = os.path.split(file)
        if basename.lower().split('.')[-1] not in ['jpg', 'png']:
            continue
        stem, ext = os.path.splitext(basename)
        gt_file = os.path.join(path, stem + '.txt')
        label_path = os.path.join(path, 'com_' + stem + '.png')
        print(label_path)
        label = cv2.imread(label_path)

        if os.path.exists(gt_file):
            with open(gt_file, 'r') as f:
                lines = f.readlines()
            boxes = []
            for line in lines:
                splitted_line = line.strip().lower().split(',')
                boxes.append(splitted_line[0:-1])
            print(np.shape(boxes))
        compare(boxes, label, img, file)
