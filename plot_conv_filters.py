import logging
import mxnet as mx
from mxboard import SummaryWriter
from mxboard.summary import *

logging.basicConfig(level=logging.INFO)

sw = SummaryWriter(logdir='/Users/jwum/Development/sandbox/logs')


def rescale(x):
    min_val = x.min().asscalar()
    max_val = x.max().asscalar()
    return (x - min_val) / (max_val - min_val)


def plot_filter(model_name, filter_name, weight):
    tag = model_name + '_' + filter_name
    sw.add_image(tag=tag + '_image', image=rescale(weight))
    sw.add_histogram(tag=tag + '_hist', values=weight, bins=100)


model_names = ['resnet_152',
               'inception_bn',
               'vgg16']
param_file_names = ['./data/resnet_152_conv0_weight.param',
                    './data/inception_bn_conv_1_weight.param',
                    './data/vgg16_conv1_1_weight.param']
filter_names = ['conv0_weight',
                'conv_1_weight',
                'conv1_1_weight']

for i, param_file in enumerate(param_file_names):
    weight = mx.nd.load(param_file)[0]
    plot_filter(model_names[i], filter_names[i], weight)

sw.close()
