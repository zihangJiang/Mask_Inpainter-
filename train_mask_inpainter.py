import argparse

import keras.backend as K
import numpy as np
import tensorflow as tf

from net import Net
from preprocess import DataFeeder
from mask_inpainter import MaskInpainter
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
parser = argparse.ArgumentParser()

#parser.add_argument('--load_img_dir', type=str, default='F:/dataset')
#parser.add_argument('--load_mask_dir', type=str, default='F:/masks')
parser.add_argument('--load_img_dir', type=str, default='/data/anaconda/Mask_Inpainting/dataset')
parser.add_argument('--load_mask_dir', type=str, default='/data/anaconda/Mask_Inpainting/mask')
parser.add_argument('--batch_size', type=int, default=20)
parser.add_argument('--image_size', type=int, default=128)


args = parser.parse_args()
load_img_dir = args.load_img_dir
load_mask_dir = args.load_mask_dir
batch_size = args.batch_size
size = (args.image_size, args.image_size)

net = Net(size[0])
data_feeder = DataFeeder(load_img_dir, load_mask_dir, batch_size=batch_size, size=size)
sess = tf.Session()
Mi = MaskInpainter(net, data_feeder, sess, batch_size, size[0],load = True)
i=0

for i in range(50):
    Mi.train(200)
    print('total epoch:{}'.format(i))
    sess.run(Mi.learning_rate_decay_op)
    Mi.generate_image('result'+str(i), concat=True)

Mi.save_models('Mi')
