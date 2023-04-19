import os

import imageio as iio
import numpy as np
import tifffile

from splicing_functions import splicing_functions
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='run Haar Wavelet blur detection with SVD on an image')
    parser.add_argument('-i', '--input_background', dest='input_background', type=str, required=True, help='image input background')
    parser.add_argument('-o', '--output_path', dest='output_path', type=str, default='output/', required=False, help="path to save output")
    parser.add_argument('-s', '--splicing_source', dest='splicing_source', type=str, help="splicing source")
    parser.add_argument('-m', '--splicing_mask', dest='splicing_mask', type=str, help="splicing mask")
    args = parser.parse_args()

    # Load image and add clouds to an image
    background = iio.imread(args.input_background).astype(np.float32)
    source = iio.imread(args.splicing_source).astype(np.float32)
    mask = tifffile.imread(args.splicing_mask).astype(np.float32)
    cloudy_image, mask = splicing_functions.splice_cloud(background, source, mask)

    # Create visual blobs
    # background_8bits = splicing_functions.convert_float32_to_uint8(background)
    # source_8bits = splicing_functions.convert_float32_to_uint8(source)
    # iio.imwrite('input/bg_8bit.png', background_8bits)
    # iio.imwrite('input/source_8bit.png', source_8bits)
    # iio.imwrite('input/mask.png', mask*255)

    # Save output
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    # Convert float32 to uint8 for IPOL
    cloudy_image_8bits = splicing_functions.convert_float32_to_uint8(cloudy_image)
    background_8bits = splicing_functions.convert_float32_to_uint8(background)
    source_8bits = splicing_functions.convert_float32_to_uint8(source)
    mask = mask * 255

    # Save 8 bits for display in IPOL
    tifffile.imwrite('output/cloudy.png', cloudy_image_8bits.astype(np.uint8))
    tifffile.imwrite('output/background.png', background_8bits.astype(np.uint8))
    tifffile.imwrite('output/mask.png', mask.astype(np.uint8))
    tifffile.imwrite('output/source.png', source_8bits.astype(np.uint8))

    # Save 16 bits for download in IPOL
    tifffile.imwrite('output/cloudy_16bits.tif', cloudy_image.astype(np.uint16))
    tifffile.imwrite('output/background_16bits.tif', background.astype(np.uint16))
    tifffile.imwrite('output/source_16bits.tif', source.astype(np.uint16))