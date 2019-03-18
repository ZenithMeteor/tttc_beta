# -*- coding: utf-8 -*-
import numpy as np
import os, sys, cv2
import glob
import shutil 

test_imgset = 'icdar_2015_val'
# ffolder = ['/img/','/pred', '/result2/', '/gt/']
ffolder = ['/img/','/2015_img_DeepLabV3_plus_2015_FC-DenseNet_v1', '/result_2015_img_DeepLabV3_plus_2015_FC-DenseNet_v1/', '/gt/']
# root = '/home/msp/Downloads/zadays/Network_zoo/draw_frame/'
root = os.getcwd() + "/"
ORIGIN = root + test_imgset + ffolder[0]
pred_DIR = root + test_imgset + ffolder[1]
RESULT_DIR = root + test_imgset + ffolder[2]
GT_DIR = root + test_imgset + ffolder[3]
if not os.path.exists(RESULT_DIR):
    os.makedirs(RESULT_DIR)

def counter(image_name):
    image = cv2.imread(image_name)
    base_name = image_name.split('/')[-1]
    file_name = base_name[5:-4]
    # im_format = base_name[-4:]
    im_format = '.jpg'
    
    image_size = image.shape
    w = image_size[1]
    h = image_size[0]
    
    # file_name = file_name.split('_')[1] + '_' + file_name.split('_')[-1]
    _im = os.path.join(ORIGIN, file_name+im_format) # _im = ORIGIN + file_name + im_format
    print(_im)
    __image = cv2.imread(_im)
    __image_size = __image.shape
    __w = __image_size[1]
    __h = __image_size[0]
    scale_w = __w/w
    scale_h = __h/h
    # re_image = cv2.resize(image, (__w, __h), interpolation=cv2.INTER_CUBIC)
    re_image = cv2.resize(__image, (w, h), interpolation=cv2.INTER_CUBIC)
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.blur(gray, (3,3))
    element = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3))
    element1 = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3))
    erode = cv2.erode(gray, element)
    dilate = cv2.dilate(erode, element1)
    #gray = dilate
    gray = cv2.Canny(dilate, 20, 150) # Canny边缘检测

    (_, cnts, _) = cv2.findContours(gray.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # clone = re_image.copy()
    clone = __image.copy()
    area = []
    # for c in cnts:
        # area.append(cv2.contourArea(c))  #計算面積
    # area_indices=np.argsort(area)
    # resize back
    clone = cv2.resize(clone, (__w, __h), interpolation=cv2.INTER_CUBIC)
    with open(GT_DIR + 'res_{}.txt'.format(file_name), 'w') as f:
        for c in cnts:
            # M = cv2.moments(c)
            # cX = int(M['m10'] / M['m00'])
            # cY = int(M['m01'] / M['m00'])        
            # 在中心點畫上黃色實心圓
            # cv2.circle(clone, (cX, cY), 10, (1, 227, 254), -1)
            
            # 擬合的旋轉矩形外框
            box = cv2.minAreaRect(c)    
            # Here, bounding rectangle is drawn with minimum area, so it considers the rotation also. 
            # The function used is cv2.minAreaRect(). It returns a Box2D structure which contains following 
            # detals - ( center (x,y), (width, height), angle of rotation ). 
            # But to draw this rectangle, we need 4 corners of the rectangle. 
            # It is obtained by the function cv2.boxPoints()
            box = np.int0(cv2.boxPoints(box))  #–> int0會省略小數點後方的數字
            # print(box)
            line, box = write_gt_txt(box, scale_w, scale_h)
            f.write(line)
            cv2.drawContours(clone, [box], -1, (0, 255, 0), 2)
    cv2.imwrite(os.path.join(RESULT_DIR + base_name), clone)
    # cv2.imwrite(os.path.join(RESULT_DIR + 'canny/' + base_name), gray)
    # cv2.imwrite(os.path.join(RESULT_DIR + 'dilate/' + base_name), erode)
    print(file_name, 'done!')
def write_gt_txt(box, scale_w, scale_h): #box:[left-down, left-up, right-up, right-down]
    box = np.ravel(box) # 将数组变为一维
    for i,b in enumerate(box):
        if i%2 == 0:
            box[i] = np.int0(b*scale_w)
        elif i%2 == 1:
            box[i] = np.int0(b*scale_h)
    # print(box)
    line = ','.join([str(box[2]),str(box[3]),str(box[4]),str(box[5])
                    ,str(box[6]),str(box[7]),str(box[0]),str(box[1])])+'\r\n'
    box = box.reshape(4,2)
    return line, box
    
def drawfcn(image_name):
    image = cv2.imread(image_name)
    base_name = image_name.split('/')[-1]
    file_name = base_name[:-9]
    im_format = base_name[-4:]
    
    image_size = image.shape
    w = image_size[1]
    h = image_size[0]
    
    _im = ORIGIN + file_name + im_format
    print(_im)
    __image = cv2.imread(_im)
    __image_size = __image.shape
    __w = __image_size[1]
    __h = __image_size[0]
    
    # re_image = cv2.resize(image, (__w, __h), interpolation=cv2.INTER_CUBIC)
    resize_image = cv2.resize(__image, (w, h), interpolation=cv2.INTER_CUBIC)
    
    for i in range(w):
        for j in range(h):
            # print(image[i][j])
            if image[i][j][0] == 100:
                resize_image[i][j] = tuple([(k+255)/2 for k in resize_image[i][j]])
                # resize_image[i][j] = (resize_image[i][j] + 100)/2
            else:
                resize_image[i][j] = tuple([(k+255)/4 for k in resize_image[i][j]])
    cv2.imwrite(os.path.join(RESULT_DIR + 'compare/' + base_name), resize_image)
    
if __name__ == '__main__':
    print(pred_DIR)
    im_names = glob.glob(os.path.join(pred_DIR, '*.png')) + \
               glob.glob(os.path.join(pred_DIR, '*.jpg'))

    # im_names = im_names[:1]
    for im_name in im_names:
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print(('Demo for {:s}'.format(im_name)))
        counter(im_name)
        # drawfcn(im_name)
    shutil.make_archive("gt", "zip", GT_DIR)