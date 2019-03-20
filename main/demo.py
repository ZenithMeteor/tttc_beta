# coding=utf-8
import os
import shutil
import sys
import time

import cv2
import numpy as np
import tensorflow as tf

sys.path.append(os.getcwd())
from nets import model_train as model
from utils.rpn_msr.proposal_layer import proposal_layer
from utils.text_connector.detectors import TextDetector
from utils import helpers

tf.app.flags.DEFINE_string('test', 'demo2/', '')
tf.app.flags.DEFINE_string('output', 'res22/', '')
tf.app.flags.DEFINE_string('gpu', '0', '')
tf.app.flags.DEFINE_string('checkpoint_path', 'checkpoints_mlt/', '')
tf.app.flags.DEFINE_boolean('draw', False, 'draw anchor')
# tf.app.flags.DEFINE_string('ver', 'v1', 'set version number')
FLAGS = tf.app.flags.FLAGS

def checkpath():
    test_data_path = os.path.join('data', FLAGS.test)
    results_path = os.path.join('data', FLAGS.output)

    if os.path.exists(results_path):
        shutil.rmtree(results_path)
    os.makedirs(results_path)

    return test_data_path, results_path

def get_images(test_data_path):
    files = []
    exts = ['jpg', 'png', 'jpeg', 'JPG']
    for parent, dirnames, filenames in os.walk(test_data_path):
        for filename in filenames:
            for ext in exts:
                if filename.endswith(ext):
                    files.append(os.path.join(parent, filename))
                    break
    print('Find {} images'.format(len(files)))
    return files


def resize_image(img):
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

    re_im = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return re_im, (new_h / img_size[0], new_w / img_size[1])


def main(argv=None):
    test_data_path, results_path = checkpath()
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

    class_names_list, label_values = helpers.get_label_info(os.path.join('./data/dataset/', "class_dict.csv"))
    num_classes = len(label_values)

    with tf.get_default_graph().as_default():
        input_image = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_image')
        input_im_info = tf.placeholder(tf.float32, shape=[None, 3], name='input_im_info')

        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

        bbox_pred, cls_pred, cls_prob, deep_pred, _ = model.model_z8(input_image, 'ResNet101', k_mode=True)

        variable_averages = tf.train.ExponentialMovingAverage(0.997, global_step)
        saver = tf.train.Saver(variable_averages.variables_to_restore())

        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            ckpt_state = tf.train.get_checkpoint_state(FLAGS.checkpoint_path)
            model_path = os.path.join(FLAGS.checkpoint_path, os.path.basename(ckpt_state.model_checkpoint_path))
            print('Restore from {}'.format(model_path))
            saver.restore(sess, model_path)

            im_fn_list = get_images(test_data_path)
            for im_fn in im_fn_list:
                print('===============')
                print(im_fn)
                start = time.time()
                try:
                    im = cv2.imread(im_fn)[:, :, ::-1]
                except:
                    print("Error reading image {}!".format(im_fn))
                    continue

                img, (rh, rw) = resize_image(im)
                h, w, c = img.shape
                im_info = np.array([h, w, c]).reshape([1, 3])
                bbox_pred_val, cls_prob_val, deep_pred_val = sess.run([bbox_pred, cls_prob, deep_pred],
                                                       feed_dict={input_image: [img],
                                                                  input_im_info: im_info})

                '''
                DeepLabV3_plus part
                '''
                _, fn = os.path.split(im_fn)
                fn, _ = os.path.splitext(fn)
                se_img = img.copy()
                output_image = np.array(deep_pred_val[0,:,:,:])
                # print(np.shape(output_image))
                output_image = helpers.reverse_one_hot(output_image)
                out_vis_image = helpers.colour_code_segmentation(output_image, label_values)
                re_pre = cv2.resize(se_img, None, None, fx=1.0 / rh, fy=1.0 / rw, interpolation=cv2.INTER_LINEAR)
                cv2.imwrite(os.path.join(results_path, 'pred_' + fn + '.png'), out_vis_image)
                # print(np.shape(out_vis_image))
                # print(w,h,fn)
                for i in range(h):
                    for j in range(w):
                        if out_vis_image[i][j][0] == 100:
                            se_img[i][j] = tuple([(k+255)/2 for k in se_img[i][j]])
                se_img = cv2.resize(se_img, None, None, fx=1.0 / rh, fy=1.0 / rw, interpolation=cv2.INTER_LINEAR)
                cv2.imwrite(os.path.join(results_path, 'com_' + fn + '.png'), se_img[:, :, ::-1]) #BGR to RGB
                '''
                cptn part
                '''
                textsegs, _ = proposal_layer(cls_prob_val, bbox_pred_val, im_info, _feat_stride=[8, ], anchor_scales=[8, ])
                scores = textsegs[:, 0]
                textsegs = textsegs[:, 1:5]

                textdetector = TextDetector(DETECT_MODE='H')
                boxes = textdetector.detect(textsegs, scores[:, np.newaxis], img.shape[:2], img, FLAGS.draw)
                boxes = np.array(boxes, dtype=np.int)
                # print(boxes)
                cost_time = (time.time() - start)
                print("cost time: {:.2f}s".format(cost_time))

                for i, box in enumerate(boxes):
                    cv2.polylines(img, [box[:8].astype(np.int32).reshape((-1, 1, 2))], True, color=(0, 255, 0),
                                  thickness=2)
                img = cv2.resize(img, None, None, fx=1.0 / rh, fy=1.0 / rw, interpolation=cv2.INTER_LINEAR)
                cv2.imwrite(os.path.join(results_path, os.path.basename(im_fn)), img[:, :, ::-1])

                with open(os.path.join(results_path, 'res_'+os.path.splitext(os.path.basename(im_fn))[0]) + ".txt", "w") as f:
                    for i, box in enumerate(boxes):
                        for k in range(8):
                            # print(box[k], k)
                            box[k] = (lambda a,b : a/rh if(b%2==0) else a/rw)(box[k], k)
                            # if k % 2 == 0:
                            #     box[k] = box[k] / rh
                            # else:
                            #     box[k] = box[k] / rw
                        # line = ",".join(str(box[k]) for k in range(8))
                        # line = ",".join(str(box[0]),str(box[1]),str(box[4]),str(box[5]))
                        line = str(box[0])+","+str(box[1])+","+str(box[4])+","+str(box[5])
                        line += "," + str(scores[i]) + "\r\n"
                        f.writelines(line)


if __name__ == '__main__':
    tf.app.run()
