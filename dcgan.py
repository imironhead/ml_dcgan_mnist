"""
Replicate DCGAN on MNIST.
arXiv:1511.06434v2
Unsupervised Representation Learning with Deep Convolutional Generative
Adversarial Networks
"""
import numpy as np
import os
import scipy.misc

import input_data

from model import Dcgan


def make_dir(dir_path):
    """
    Helper function to make a directory if it doesn`t exist.
    """
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)


def next_real_batch(params):
    """
    Get next batch from MINST. A bunch of 32x32 mono channel images as Numpy
    arrays. Shape of the batched data is [batch_size, 32, 32, 1].
    """
    if 'mnist' not in params:
        # lazy loading.
        path_dir_mnist = params.get('path_dir_mnist', './data/')

        params['mnist'] = input_data.read_data_sets(path_dir_mnist)

    mnist = params['mnist']

    discriminator_batch_size = params.get('discriminator_batch_size', 128)

    discriminator_batch, _ = mnist.train.next_batch(discriminator_batch_size)

    # discriminator_batch.shape: [discriminator_batch_size, 784]

    # reshape into [discriminator_batch_size, 28, 28, 1]
    # width = height = 28, channel = 1
    discriminator_batch = np.reshape(discriminator_batch, [-1, 28, 28, 1])

    # map pixel value from (0.0, 1.0) to (-1.0, +1.0)
    discriminator_batch = 2.0 * (discriminator_batch - 0.5)

    # pad to 32 * 32 images with -1.0
    discriminator_batch = np.pad(
        discriminator_batch,
        ((0, 0), (2, 2), (2, 2), (0, 0)),
        'constant',
        constant_values=(-1.0, -1.0))

    return discriminator_batch


def next_fake_batch(params):
    """
    Return random seeds for the generator.
    """
    generator_batch_size = params.get('generator_batch_size', 128)

    generator_seed_size = params.get('generator_seed_size', 128)

    batch = np.random.uniform(
        -1.0,
        1.0,
        size=[generator_batch_size, generator_seed_size])

    return batch.astype(np.float32)


def save_merged_results(params, results, path_merged_results):
    """
    Save the generated images into a sprite sheet.
    """
    if len(results.shape) > 3:
        # remove the rank of image channel.
        # [-1, 32, 32, 1] to [-1, 32, 32]
        results = np.squeeze(results, axis=(3,))

    width, height = results.shape[1:3]

    count_x = params.get('results_image_x_count', 16)
    count_y = params.get('results_image_y_count', 16)

    image = np.zeros((height * count_y, width * count_x))

    results = 0.5 * (results + 1.0)

    # make the sheet.
    for idx, result in enumerate(results):
        x = (idx % count_x) * width
        y = (idx / count_x) * height

        image[y:y + height, x:x + width] = result

    scipy.misc.imsave(path_merged_results, image)


def train(params):
    """
    """
    print 'training'

    # create model
    dcgan = Dcgan(params)

    # train
    for iteration in xrange(params['training_iterations']):
        real_sources = next_real_batch(params)
        fake_sources = next_fake_batch(params)

        dcgan.train_discriminator(fake_sources, real_sources)
        dcgan.train_generator(fake_sources)
        dcgan.train_generator(fake_sources)

        print 'iteration: {}'.format(iteration)

        # peek the generator.
        if iteration % 10 == 0:
            fake_results = dcgan.generate(fake_sources)

            path_dir_results = params.get('path_dir_results', './results/')

            path_results = os.path.join(
                path_dir_results, 'training_{}.png'.format(iteration))

            save_merged_results(params, fake_results, path_results)


if __name__ == '__main__':
    # training parameters
    params = {}
    params['path_dir_mnist'] = './data/'
    params['path_dir_results'] = './results/'
    params['training_iterations'] = 10000
    params['discriminator_batch_size'] = 128
    params['generator_batch_size'] = 128
    params['generator_seed_size'] = 128
    params['results_image_x_count'] = 16
    params['results_image_y_count'] = 16

    make_dir(params['path_dir_mnist'])
    make_dir(params['path_dir_results'])

    train(params)
