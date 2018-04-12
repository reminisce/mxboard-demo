import logging
import mxnet as mx
from mxboard import SummaryWriter


logging.basicConfig(level=logging.INFO)
resized_images = mx.nd.load('imagenet1k-resnet-152_resized_images.ndarray')[0]
convnet_codes = mx.nd.load('imagenet1k-resnet-152_convnet_codes.ndarray')[0]

with SummaryWriter(logdir='./logs') as sw:
    sw.add_embedding(tag='resnet_152_image_codes', embedding=convnet_codes, images=resized_images,
                     labels=['N/A' for _ in range(convnet_codes.shape[0])])
