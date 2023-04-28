import numpy as np
import iio
import os


def translation_mask(mask, dx, dy, shape_out):
    h, w = mask.shape[:2]
    dx = int(dx)
    dy = int(dy)
    ts_mat = np.array([[1, 0, dx], [0, 1, dy]])

    translated_mask = np.zeros(shape_out)
    for i in range(h):
        for j in range(w):
            origin_x = j
            origin_y = i
            origin_xy = np.array([origin_x, origin_y, 1])

            new_xy = np.dot(ts_mat, origin_xy)
            new_x = new_xy[0]
            new_y = new_xy[1]

            if 0 < new_x < w and 0 < new_y < h:
                translated_mask[new_y, new_x] = mask[i, j]

    return translated_mask


def generate_perlin_noise_2d(shape, res):
    def f(t):
        return 6 * t ** 5 - 15 * t ** 4 + 10 * t ** 3

    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])
    grid = np.mgrid[0:res[0]:delta[0], 0:res[1]:delta[1]].transpose(1, 2, 0) % 1
    # Gradients
    angles = 2 * np.pi * np.random.rand(int(res[0]) + 1, int(res[1]) + 1)
    gradients = np.dstack((np.cos(angles), np.sin(angles)))
    g00 = gradients[0:-1, 0:-1].repeat(d[0], 0).repeat(d[1], 1)
    g10 = gradients[1:, 0:-1].repeat(d[0], 0).repeat(d[1], 1)
    g01 = gradients[0:-1, 1:].repeat(d[0], 0).repeat(d[1], 1)
    g11 = gradients[1:, 1:].repeat(d[0], 0).repeat(d[1], 1)
    # Ramps
    n00 = np.sum(grid * g00, 2)
    n10 = np.sum(np.dstack((grid[:, :, 0] - 1, grid[:, :, 1])) * g10, 2)
    n01 = np.sum(np.dstack((grid[:, :, 0], grid[:, :, 1] - 1)) * g01, 2)
    n11 = np.sum(np.dstack((grid[:, :, 0] - 1, grid[:, :, 1] - 1)) * g11, 2)
    # Interpolation
    t = f(grid)
    n0 = n00 * (1 - t[:, :, 0]) + t[:, :, 0] * n10
    n1 = n01 * (1 - t[:, :, 0]) + t[:, :, 0] * n11

    return np.sqrt(2) * ((1 - t[:, :, 1]) * n0 + t[:, :, 1] * n1)


def generate_fractal_noise_2d(shape, res, octaves=1, persistence=0.5):
    noise = np.zeros(shape)
    frequency = 1
    amplitude = 1
    for _ in range(octaves):
        noise += amplitude * generate_perlin_noise_2d(shape, (frequency * res[0], frequency * res[1]))
        frequency *= 2
        amplitude *= persistence

    return noise


def cloud_generation(shape, res, octaves):
    fractal_noise = generate_fractal_noise_2d(shape, res, octaves)
    mask = (fractal_noise > 0.1) * 255

    return fractal_noise, mask


def cloud_copy(img, cloud):
    cloud_blue = cloud
    cloud_green = translation_mask(cloud, 5, 5, cloud.shape)
    cloud_red = translation_mask(cloud, 10, 10, cloud.shape)
    cloud = np.stack([cloud_red, cloud_green, cloud_blue], axis=2)

    # Crop cloud to fit image size
    cloud_crop = cloud[50:(50 + img.shape[0]), 50:50 + img.shape[1]]
    cloud_crop = (cloud_crop - np.min(cloud_crop)) / (np.max(cloud_crop) - np.min(cloud_crop))

    # noise
    # noise = np.random.normal(0, 0.01, cloud.shape)
    # cloud = cloud + noise

    cloudy = (cloud_crop * (2 ** 12)) * cloud_crop + (1 - cloud_crop) * img

    return cloudy, cloud_crop


def convert_float32_to_uint8(img):
    if len(img.shape) > 2:
        rescale = []
        for i in range(img.shape[2]):
            channel = np.sqrt(img[:, :, i]) / np.percentile(np.sqrt(img), 99) * 255
            channel[channel > 255] = 255
            rescale.append(channel)
        rescale = np.stack(rescale, axis=2)
    else:
        rescale = img / np.max(img) * 255
        rescale = rescale[:, :]

    return rescale.astype(np.uint8)


def cloud_resolution(img):
    shape = np.max(img.shape)
    powers = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]

    for power in powers:
        if power > shape:
            return power


def spliceCloudFromGenertedMask(input, res, octave):
    u = iio.read(input)
    cloud_resolution_size = cloud_resolution(u)
    cloud, mask = cloud_generation((cloud_resolution_size, cloud_resolution_size), (res, res), octave)
    cloudy, cloud_crop = cloud_copy(u, cloud)

    if not os.path.exists('output_generated/'):
        os.mkdir('output_generated/')

    # Rescale image on 8 bits for visualization
    u_8bits = convert_float32_to_uint8(u)
    cloudy_8bits = convert_float32_to_uint8(cloudy)

    # Save display outputs
    iio.write('output_generated/background.png', u_8bits)
    iio.write('output_generated/cloud.png', (cloud_crop * 255).astype(np.uint8))
    iio.write('output_generated/cloudy.png', cloudy_8bits)

    # Save download outputs
    iio.write('output_generated/background.tif', u)
    iio.write('output_generated/cloud.tif', cloud_crop)
    iio.write('output_generated/cloudy.tif', cloudy.astype(np.uint16))
