# codes from nvidia vpi tutorial
import vpi

import numpy as np
from PIL import Image
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('input', help='Image to be used as input')
args = parser.parse_args()

input = vpi.asimage(np.asarray(Image.open(args.input)))

with vpi.Backend.CUDA:
    output = input.convert(vpi.Format.U8) \
                    .box_filter(5, border=vpi.Border.ZERO)

with output.rlock():
    Image.fromarray(output.cpu()).save('step001_blurandsave_output.png')
    