import logging
import mxnet as mx
from mxboard import SummaryWriter


logging.basicConfig(level=logging.INFO)
resized_images = mx.nd.load('./data/imagenet1k-resnet-152_resized_images.ndarray')[0]
convnet_codes = mx.nd.load('./data/imagenet1k-resnet-152_convnet_codes.ndarray')[0]
labels = mx.nd.load('./data/imagenet1k-resnet-152_labels.ndarray')[0].asnumpy()


with open('./data/synset.txt', 'r') as f:
    label_strs = [l.rstrip() for l in f]

with SummaryWriter(logdir='./logs') as sw:
    sw.add_image(tag='imagenet_2304_images', image=resized_images)
    sw.add_embedding(tag='resnet_152_image_codes', embedding=convnet_codes, images=resized_images,
                     labels=[label_strs[idx][10:] for idx in labels])
