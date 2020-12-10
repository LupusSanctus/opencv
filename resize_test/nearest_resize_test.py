from __future__ import print_function

import cv2
import cv2 as cv
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.python.tools import optimize_for_inference_lib
from tensorflow.tools.graph_transforms import TransformGraph

np.random.seed(2701)

print('OpenCV:', cv.__version__)
import PIL
from PIL import Image

print('PIL:', PIL.__version__)
import skimage
from skimage.transform import resize

import tensorflow as tf

print('scikit-image:', skimage.__version__)


def prepareOutData(data):
    if data.ndim == 4:
        return data.transpose(0, 3, 1, 2).astype(int)
    else:
        return data.astype(int)


def readData(data):
    if data.ndim == 4:
        return data.transpose(0, 2, 3, 1)
    else:
        return data


sess = tf.Session()
isTraining = tf.placeholder(tf.bool, name='isTraining')


def run_tf(inp, out, data):
    sess.run(tf.global_variables_initializer())

    data = np.expand_dims(data, axis=(0, 1))
    data = readData(data)
    outputData = sess.run(out, feed_dict={inp: data, isTraining: False})
    return prepareOutData(outputData)


def resize_compare(size, new_size):
    img = np.arange(size[0] * size[1], dtype=np.uint8).reshape(size)

    print('\nimg:\n', img)

    # OpenCV
    img_resize_OpenCV = cv.resize(img, (new_size[1], new_size[0]), interpolation=cv.INTER_NEAREST_EXACT)
    print('\nOpenCV:\n', img_resize_OpenCV)

    # Pillow
    im = Image.fromarray(img)
    img_resize_PIL = im.resize((new_size[1], new_size[0]), resample=Image.NEAREST)
    print('\nPIL:\n', np.array(img_resize_PIL))

    # scikit-image
    img_resize_sk = resize(img, new_size, order=0, preserve_range=True).astype(np.uint8)
    print('\nskimage:\n', img_resize_sk)

    # TF 1.5 half_pixel
    inp = tf.placeholder(tf.float32, [1, size[0], size[1], 1], 'input')
    resized = tf.image.resize_nearest_neighbor(inp, size=(new_size[0], new_size[1]), align_corners=False,
                                               name='resize_nearest_neighbor',
                                               half_pixel_centers=True)
    tf_out1 = run_tf(inp, resized, img)
    print('\nTF half_pixel:\n', tf_out1)

    # # TF 1.5 align_corners
    # inp = tf.placeholder(tf.float32, [1, size[0], size[1], 1], 'input')
    # resized = tf.image.resize_nearest_neighbor(inp, size=(new_size[0], new_size[1]), align_corners=True,
    #                                            name='resize_nearest_neighbor',
    #                                            half_pixel_centers=False)
    # tf_out2 = run_tf(inp, resized, img)
    # print('\nTF align_corners:\n', tf_out2)
    #
    # # TF 1.5
    # inp = tf.placeholder(tf.float32, [1, size[0], size[1], 1], 'input')
    # resized = tf.image.resize_nearest_neighbor(inp, size=(new_size[0], new_size[1]), align_corners=False,
    #                                            name='resize_nearest_neighbor',
    #                                            half_pixel_centers=False)
    # tf_out3 = run_tf(inp, resized, img)
    # print('\nTF:\n', tf_out3)


# result: all matrices are different
print("\n ============== Test: 10x11 => 15x13")
size = (10, 11)
new_size = (15, 13)
resize_compare(size, new_size)

# result: almost equal outputs - except TF align_corners=False, half_pixel_centers=False
print("\n ============== Test: 10x11 => 14x12")
size = (10, 11)
new_size = (14, 12)
resize_compare(size, new_size)

## Previous tests
# size = (3, 5)
# new_size = (5, 7)
# resize_compare(size, new_size)

# size = (2, 3)
# new_size = (4, 6)
# resize_compare(size, new_size)
