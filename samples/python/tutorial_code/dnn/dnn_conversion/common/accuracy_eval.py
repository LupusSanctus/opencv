import os
import sys
import time
from abc import ABCMeta, abstractmethod

import cv2
import numpy as np
from tensorflow.keras.applications.resnet import preprocess_input


# https://github.com/opencv/opencv/blob/master/modules/dnn/test/imagenet_cls_test_alexnet.py#L98
def get_correct_answers(img_list, img_classes, net_output_blob):
    correct_answers = 0
    for i in range(len(img_list)):
        indexes = np.argsort(net_output_blob[i])[-5:]
        correct_index = img_classes[img_list[i]]
        if correct_index in indexes:
            correct_answers += 1
    return correct_answers


# https://github.com/opencv/opencv/blob/master/modules/dnn/test/imagenet_cls_test_alexnet.py#L26
class DataFetch(object):
    imgs_dir = ''
    frame_size = 0
    bgr_to_rgb = False

    __metaclass__ = ABCMeta

    @abstractmethod
    def preprocess(self, img):
        pass

    def get_batch(self, imgs_names):
        assert type(imgs_names) is list
        batch = np.zeros((len(imgs_names), 3, self.frame_size, self.frame_size)).astype(np.float32)
        for i in range(len(imgs_names)):
            img_name = imgs_names[i]
            img_file = os.path.join(self.imgs_dir, img_name)
            assert os.path.exists(img_file)
            img = cv2.imread(img_file, cv2.IMREAD_COLOR)
            min_dim = min(img.shape[-3], img.shape[-2])
            resize_ratio = self.frame_size / float(min_dim)
            img = cv2.resize(img, (0, 0), fx=resize_ratio, fy=resize_ratio)
            cols = img.shape[1]
            rows = img.shape[0]
            y1 = round((rows - self.frame_size) / 2)
            y2 = round(y1 + self.frame_size)
            x1 = round((cols - self.frame_size) / 2)
            x2 = round(x1 + self.frame_size)
            img = img[y1:y2, x1:x2]
            if self.bgr_to_rgb:
                img = img[..., ::-1]
            image_data = img[:, :, 0:3].transpose(2, 0, 1)
            batch[i] = self.preprocess(image_data)
        return batch


def normalize_imgs(img):
    image_data = np.array(img).astype(np.float32)
    image_data = np.expand_dims(image_data, 0)
    image_data /= 255.0
    return image_data


class NormalizedValueFetch(DataFetch):
    def __init__(self, imgs_dir, frame_size, bgr_to_rgb):
        self.imgs_dir = imgs_dir
        self.frame_size = frame_size
        self.bgr_to_rgb = bgr_to_rgb

    def preprocess(self, img):
        image_data = np.array(img).astype(np.float32)
        image_data = np.expand_dims(image_data, 0)
        image_data /= 255.0
        return image_data


class TFPreprocessedFetch(DataFetch):
    def __init__(self, frame_size, imgs_dir, bgr_to_rgb):
        self.imgs_dir = imgs_dir
        self.frame_size = frame_size
        self.bgr_to_rgb = bgr_to_rgb

    def preprocess(self, img):
        return preprocess_input(img)


# https://github.com/opencv/opencv/blob/master/modules/dnn/test/imagenet_cls_test_alexnet.py#L159
class ClsAccEvaluation:
    log = sys.stdout
    img_classes = {}
    batch_size = 0

    def __init__(self, log_path, img_classes_file, batch_size):
        self.log = open(log_path, 'w')
        self.img_classes = self.read_classes(img_classes_file)
        self.batch_size = batch_size
        # collect the accuracies for both models
        self.general_fw_accuracy = []

    @staticmethod
    def read_classes(img_classes_file):
        result = {}
        with open(img_classes_file) as file:
            for l in file.readlines():
                result[l.split()[0]] = int(l.split()[1])
        return result

    def process(self, frameworks, data_fetcher):
        sorted_imgs_names = sorted(self.img_classes.keys())
        correct_answers = [0] * len(frameworks)
        samples_handled = 0
        blobs_l1_diff = [0] * len(frameworks)
        blobs_l1_diff_count = [0] * len(frameworks)
        blobs_l_inf_diff = [sys.float_info.min] * len(frameworks)
        inference_time = [0.0] * len(frameworks)

        for x in range(0, len(sorted_imgs_names), self.batch_size):
            sublist = sorted_imgs_names[x:x + self.batch_size]
            batch = data_fetcher.get_batch(sublist)

            samples_handled += len(sublist)
            fw_accuracy = []
            frameworks_out = []
            for i in range(len(frameworks)):
                start = time.time()
                out = frameworks[i].get_output(batch)
                end = time.time()
                correct_answers[i] += get_correct_answers(sublist, self.img_classes, out)
                fw_accuracy.append(100 * correct_answers[i] / float(samples_handled))
                frameworks_out.append(out)
                inference_time[i] += end - start
                print(samples_handled, 'Accuracy for', frameworks[i].get_name() + ':', fw_accuracy[i], file=self.log)
                print("Inference time, ms ", frameworks[i].get_name(), inference_time[i] / samples_handled * 1000,
                      file=self.log)

                self.general_fw_accuracy.append(fw_accuracy)

            for i in range(1, len(frameworks)):
                log_str = frameworks[0].get_name() + " vs " + frameworks[i].get_name() + ':'
                diff = np.abs(frameworks_out[0] - frameworks_out[i])
                l1_diff = np.sum(diff) / diff.size
                print(samples_handled, "L1 difference", log_str, l1_diff, file=self.log)
                blobs_l1_diff[i] += l1_diff
                blobs_l1_diff_count[i] += 1
                if np.max(diff) > blobs_l_inf_diff[i]:
                    blobs_l_inf_diff[i] = np.max(diff)
                print(samples_handled, "L_INF difference", log_str, blobs_l_inf_diff[i], file=self.log)

            self.log.flush()

        for i in range(1, len(blobs_l1_diff)):
            log_str = frameworks[0].get_name() + " vs " + frameworks[i].get_name() + ':'
            print('Final l1 diff', log_str, blobs_l1_diff[i] / blobs_l1_diff_count[i], file=self.log)


# segmentation
def get_conf_mat(gt, prob):
    assert type(gt) is np.ndarray
    assert type(prob) is np.ndarray

    conf_mat = np.zeros((gt.shape[0], gt.shape[0]))
    for ch_gt in range(conf_mat.shape[0]):
        gt_channel = gt[ch_gt, ...]
        for ch_pr in range(conf_mat.shape[1]):
            prob_channel = prob[ch_pr, ...]
            conf_mat[ch_gt][ch_pr] = np.count_nonzero(np.multiply(gt_channel, prob_channel))
    return conf_mat


def eval_segm_result(net_out):
    assert type(net_out) is np.ndarray
    assert len(net_out.shape) == 4

    channels_dim = 1
    y_dim = channels_dim + 1
    x_dim = y_dim + 1
    res = np.zeros(net_out.shape).astype(np.int)
    for i in range(net_out.shape[y_dim]):
        for j in range(net_out.shape[x_dim]):
            max_ch = np.argmax(net_out[..., i, j])
            res[0, max_ch, i, j] = 1
    return res


def get_metrics(conf_mat):
    pix_accuracy = np.trace(conf_mat) / np.sum(conf_mat)
    t = np.sum(conf_mat, 1)
    num_cl = np.count_nonzero(t)
    assert num_cl
    mean_accuracy = np.sum(np.nan_to_num(np.divide(np.diagonal(conf_mat), t))) / num_cl
    col_sum = np.sum(conf_mat, 0)
    mean_iou = np.sum(
        np.nan_to_num(np.divide(np.diagonal(conf_mat), (t + col_sum - np.diagonal(conf_mat))))) / num_cl
    return pix_accuracy, mean_accuracy, mean_iou


class SemSegmEvaluation:
    log = sys.stdout

    def __init__(self, log_path, ):
        self.log = open(log_path, 'w')

    def process(self, frameworks, data_fetcher):
        samples_handled = 0

        conf_mats = [np.zeros((data_fetcher.get_num_classes(), data_fetcher.get_num_classes())) for i in
                     range(len(frameworks))]
        blobs_l1_diff = [0] * len(frameworks)
        blobs_l1_diff_count = [0] * len(frameworks)
        blobs_l_inf_diff = [sys.float_info.min] * len(frameworks)
        inference_time = [0.0] * len(frameworks)

        for in_blob, gt in data_fetcher:
            frameworks_out = []
            samples_handled += 1
            for i in range(len(frameworks)):
                start = time.time()
                out = frameworks[i].get_output(in_blob)
                end = time.time()
                segm = eval_segm_result(out)
                conf_mats[i] += get_conf_mat(gt, segm[0])
                frameworks_out.append(out)
                inference_time[i] += end - start

                pix_acc, mean_acc, miou = get_metrics(conf_mats[i])

                name = frameworks[i].get_name()
                print(samples_handled, 'Pixel accuracy, %s:' % name, 100 * pix_acc, file=self.log)
                print(samples_handled, 'Mean accuracy, %s:' % name, 100 * mean_acc, file=self.log)
                print(samples_handled, 'Mean IOU, %s:' % name, 100 * miou, file=self.log)
                print("Inference time, ms ", \
                      frameworks[i].get_name(), inference_time[i] / samples_handled * 1000, file=self.log)

            for i in range(1, len(frameworks)):
                log_str = frameworks[0].get_name() + " vs " + frameworks[i].get_name() + ':'
                diff = np.abs(frameworks_out[0] - frameworks_out[i])
                l1_diff = np.sum(diff) / diff.size
                print(samples_handled, "L1 difference", log_str, l1_diff, file=self.log)
                blobs_l1_diff[i] += l1_diff
                blobs_l1_diff_count[i] += 1
                if np.max(diff) > blobs_l_inf_diff[i]:
                    blobs_l_inf_diff[i] = np.max(diff)
                print(samples_handled, "L_INF difference", log_str, blobs_l_inf_diff[i], file=self.log)

            self.log.flush()

        for i in range(1, len(blobs_l1_diff)):
            log_str = frameworks[0].get_name() + " vs " + frameworks[i].get_name() + ':'
            print('Final l1 diff', log_str, blobs_l1_diff[i] / blobs_l1_diff_count[i], file=self.log)


class DatasetImageFetch(object):
    __metaclass__ = ABCMeta
    data_prepoc = object

    @abstractmethod
    def __iter__(self):
        pass

    @abstractmethod
    def next(self):
        pass

    @staticmethod
    def pix_to_c(pix):
        return pix[0] * 256 * 256 + pix[1] * 256 + pix[2]

    @staticmethod
    def color_to_gt(color_img, colors):
        num_classes = len(colors)
        gt = np.zeros((num_classes, color_img.shape[0], color_img.shape[1])).astype(np.int)
        for img_y in range(color_img.shape[0]):
            for img_x in range(color_img.shape[1]):
                c = DatasetImageFetch.pix_to_c(color_img[img_y][img_x])
                if c in colors:
                    cls = colors.index(c)
                    gt[cls][img_y][img_x] = 1
        return gt


class PASCALDataFetch(DatasetImageFetch):
    img_dir = ''
    segm_dir = ''
    names = []
    colors = []
    i = 0

    def __init__(self, img_dir, segm_dir, names_file, segm_cls_colors_file, preproc):
        self.img_dir = img_dir
        self.segm_dir = segm_dir
        self.colors = self.read_colors(segm_cls_colors_file)
        self.data_prepoc = preproc
        self.i = 0

        with open(names_file) as f:
            for l in f.readlines():
                self.names.append(l.rstrip())

    @staticmethod
    def read_colors(img_classes_file):
        result = []
        with open(img_classes_file) as f:
            for l in f.readlines():
                color = np.array(map(int, l.split()[1:]))
                result.append(DatasetImageFetch.pix_to_c(color))
        return result

    def __iter__(self):
        return self

    def next(self):
        if self.i < len(self.names):
            name = self.names[self.i]
            self.i += 1
            segm_file = self.segm_dir + name + ".png"
            img_file = self.img_dir + name + ".jpg"
            gt = self.color_to_gt(cv2.imread(segm_file, cv2.IMREAD_COLOR)[:, :, ::-1], self.colors)
            img = self.data_prepoc(cv2.imread(img_file, cv2.IMREAD_COLOR)[:, :, ::-1])
            return img, gt
        else:
            self.i = 0
            raise StopIteration

    def get_num_classes(self):
        return len(self.colors)
