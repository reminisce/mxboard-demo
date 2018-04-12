import logging
import mxnet as mx
from mxboard import SummaryWriter

logging.basicConfig(level=logging.INFO)


def rescale(x, x_min=None, x_max=None):
    if x_min is None:
        x_min = x.min().asscalar()
    if x_max is None:
        x_max = x.max().asscalar()
    return (x - x_min) / (x_max - x_min)


def rescale_per_image(x):
    assert x.ndim == 4
    x = x.copy()
    for i in range(x.shape[0]):
        min_val = x[i].min().asscalar()
        max_val = x[i].max().asscalar()
        x[i] = rescale(x[i], min_val, max_val)
    return x


sw = SummaryWriter(logdir='./logs')

swan = mx.nd.load('./data/imagenet_swan.ndarray')[0]
swan = swan.reshape((1,) + swan.shape).astype('float32')
sw.add_image(tag='swan', image=swan.astype('uint8'))

# plot conv filter and output of inception-bn
weight = mx.nd.load('./data/inception_bn_conv_1_weight.param')[0]
bias = mx.nd.load('./data/inception_bn_conv_1_bias.param')[0]
out = mx.nd.Convolution(swan, weight=weight, bias=bias, kernel=weight.shape[2:], num_filter=weight.shape[0])
out = out.transpose((1, 0, 2, 3))
tag = 'test_weight'
sw.add_image(tag='inception_bn_conv_1_weight', image=rescale_per_image(weight))
sw.add_image(tag='inception_bn_conv_1_output', image=rescale_per_image(out))

# plot conv filter and output of resnet-152
weight = mx.nd.load('./data/resnet_152_conv0_weight.param')[0]
out = mx.nd.Convolution(swan, weight=weight, kernel=weight.shape[2:], num_filter=weight.shape[0], no_bias=True)
out = out.transpose((1, 0, 2, 3))
sw.add_image(tag='resnet_152_conv0_weight', image=rescale_per_image(weight))
sw.add_image(tag='resnet_152_conv0_output', image=rescale_per_image(out))

# plot conv filter and output of vgg16
weight = mx.nd.load('./data/vgg16_conv1_1_weight.param')[0]
bias = mx.nd.load('./data/vgg16_conv1_1_bias.param')[0]
out = mx.nd.Convolution(swan, weight=weight, bias=bias, kernel=weight.shape[2:], num_filter=weight.shape[0])
out = out.transpose((1, 0, 2, 3))
sw.add_image(tag='vgg16_conv1_1_weight', image=rescale_per_image(weight))
sw.add_image(tag='vgg16_conv1_1_output', image=rescale_per_image(out))

sw.close()
