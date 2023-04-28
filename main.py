from splicing_functions import splicing_functions
from CloudPerlin import CloudPerlin
import argparse


def main(background, source, mask, conv_size, radius, epsilon, res, octave, mode):
    if mode == 0:  # Splice cloud from a real one
        splicing_functions.spliceCloudFromMask(background, source, mask, conv_size, radius, epsilon)
    elif mode == 1:  # Fully generated cloud
        CloudPerlin.spliceCloudFromGenertedMask(background, res, octave)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='run Haar Wavelet blur detection with SVD on an image')
    parser.add_argument('-i', '--input_background', dest='input_background', type=str, required=True,
                        help='image input background')
    parser.add_argument('-s', '--splicing_source', dest='splicing_source', type=str, help="splicing source")
    parser.add_argument('-mask', '--splicing_mask', dest='splicing_mask', type=str, help="splicing mask")
    parser.add_argument('-c', '--conv_size', dest='conv_size', type=int, default=5, help="Convolution size")
    parser.add_argument('-rad', '--radius', dest='radius', type=int, default=16, help="Radius")
    parser.add_argument('-e', '--epsilon', dest='epsilon', type=float, default=0.01, help="Epsilon")
    parser.add_argument('-o', '--octave', dest='octave', type=int, default=7, help="octave noise parameter")
    parser.add_argument('-r', '--res', dest='res', type=int, default=2, help="res noise parameter")
    parser.add_argument('-mode', '--mode', dest='mode', type=int, default=1, help="which mode you want to use")
    args = parser.parse_args()

    main(args.input_background, args.splicing_source, args.splicing_mask, args.conv_size, args.radius, args.epsilon,
         args.res, args.octave, args.mode)
