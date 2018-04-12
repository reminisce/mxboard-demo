import os
import argparse
import logging
import mxnet as mx
from common import modelzoo
from mxnet.module import Module


def download_dataset(dataset_url, dataset_path):
    logging.info('Downloading dataset from %s to %s' % (dataset_url, dataset_path))
    mx.test_utils.download(dataset_url, dataset_path)


def download_model(model_name):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    model_path = os.path.join(dir_path, 'model')
    logging.info('Downloading model %s... into path %s' % (model_name, model_path))
    return modelzoo.download_model(args.model, os.path.join(dir_path, 'model'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate a calibrated quantized model from a FP32 model')
    parser.add_argument('--max-num-images', type=int, default=2304, help='max-num-images-collected')
    parser.add_argument('--model', type=str, default='imagenet1k-inception-bn',
                        choices=['imagenet1k-resnet-152', 'imagenet1k-inception-bn'],
                        help='currently only supports imagenet1k-resnet-152 or imagenet1k-inception-bn')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    dataset_path = 'data/val_256_q90.rec'
    download_dataset('http://data.mxnet.io/data/val_256_q90.rec', dataset_path)
    prefix, epoch = download_model(model_name=args.model)
    sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
    fc_output_name = 'fc1_output'
    convnet_code_sym = sym.get_internals()[fc_output_name]

    ctx = mx.gpu(0)
    data_nthreads = 60
    batch_size = 8
    data_shape = (3, 224, 224)
    label_name = 'softmax_label'
    shuffle_dataset = True
    shuffle_chunk_seed = 3982304
    shuffle_seed = 48564309
    if args.model == 'imagenet1k-resnet-152':
        mean_args = {'mean_r': 0, 'mean_g': 0, 'mean_b': 0}
    elif args.model == 'imagenet1k-inception-bn':
        mean_args = {'mean_r': 123.68, 'mean_g': 116.779, 'mean_b': 103.939}
    else:
        raise ValueError('unsupported model %s' % args.model)

    data = mx.io.ImageRecordIter(path_imgrec=dataset_path,
                                 label_width=1,
                                 preprocess_threads=data_nthreads,
                                 batch_size=batch_size,
                                 data_shape=data_shape,
                                 label_name=label_name,
                                 rand_crop=False,
                                 rand_mirror=False,
                                 shuffle=shuffle_dataset,
                                 shuffle_chunk_seed=shuffle_seed,
                                 seed=shuffle_seed,
                                 **mean_args)
    mod = Module(symbol=convnet_code_sym, label_names=None, context=ctx)
    mod.bind(for_training=False, data_shapes=data.provide_data)
    mod.set_params(arg_params, aux_params)
    num_images = 0
    convnet_codes = None  # N * 1000
    resized_images = None  # NCHW
    for batch in data:
        if num_images >= args.max_num_images:
            break
        mod.forward(data_batch=batch, is_train=False)
        fc_output = mod.get_outputs()[0].flatten().copyto(mx.cpu(0))
        num_images += batch_size
        fc_output.wait_to_read()
        if convnet_codes is None:
            convnet_codes = fc_output
        else:
            convnet_codes = mx.nd.concat(*[convnet_codes, fc_output], dim=0)
        images = batch.data[0].copyto(mx.cpu(0))  # batch images in NCHW
        images = images.transpose((0, 2, 3, 1))  # batch images in NHWC
        images.wait_to_read()
        for i in range(images.shape[0]):
            resized_image = mx.img.resize_short(images[i], size=64).transpose((2, 0, 1)).expand_dims(axis=0)  # NCHW
            resized_image[0][0] += mean_args['mean_r']
            resized_image[0][1] += mean_args['mean_g']
            resized_image[0][2] += mean_args['mean_b']
            resized_image = mx.nd.clip(resized_image, 0, 255).astype('uint8')
            if resized_images is None:
                resized_images = resized_image
            else:
                resized_images = mx.nd.concat(*[resized_images, resized_image], dim=0)
        logging.info('collected %d images and convnet codes so far' % num_images)
        mx.nd.waitall()
    mx.nd.save('%s_convnet_codes.ndarray' % args.model, convnet_codes)
    mx.nd.save('%s_resized_images.ndarray' % args.model, resized_images)
