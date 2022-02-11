import jetson.inference
import jetson.utils

import numpy as np
import vpi
from PIL import Image

input0 = jetson.utils.videoSource("csi://0")
input1 = jetson.utils.videoSource("csi://1")
img0 = input0.Capture()
img1 = input1.Capture()
disp_img = Image.new('L', (img0.width + img1.width, img0.height))
overlayText0 = jetson.utils.cudaFont()
overlayText1 = jetson.utils.cudaFont()
output = jetson.utils.videoOutput("display://0")

captureLength = 30*60
frameNo = 0

while True:
    img0 = input0.Capture()
    img1 = input1.Capture()
    frameNo += 1

    vpi_img0 = vpi.asimage(np.asarray(img0))
    vpi_img1 = vpi.asimage(np.asarray(img1))

    with vpi.Backend.CUDA:
        p1_vpiimg0 = vpi_img0.convert(vpi.Format.U8) \
                                .box_filter(5, border=vpi.Border.ZERO) \
                                .rescale((vpi_img0.width*0.5, vpi_img0.height*0.5), interp=vpi.Interp.LINEAR, border=vpi.Border.ZERO) \
                                .box_filter(5, border=vpi.Border.ZERO)
        p1_vpiimg1 = vpi_img1.convert(vpi.Format.U8) \
                                .box_filter(5, border=vpi.Border.ZERO) \
                                .rescale((int(vpi_img1.width*0.5), int(vpi_img1.height*0.5)), interp=vpi.Interp.LINEAR, border=vpi.Border.ZERO) \
                                .box_filter(5, border=vpi.Border.ZERO)
    with p1_vpiimg0.rlock():
        out_img0 = p1_vpiimg0.cpu()
    with p1_vpiimg1.rlock():
        out_img1 = p1_vpiimg1.cpu()

    overlayText0.OverlayText(out_img0, out_img0.width, out_img0.height, "csi:0, Gray, Box5x5, wx0.5&hx0.5, Box5x5", 2, 2, overlayText0.White, overlayText0.Gray60)
    overlayText1.OverlayText(out_img1, out_img1.width, out_img1.height, "csi:1, Gray, Box5x5, wx0.5&hx0.5, Box5x5", 2, 2, overlayText1.White, overlayText1.Gray60)

    disp_img.paste(out_img0, (0,0))
    disp_img.paste(out_img1, (out_img0.width, 0))

    output.Render(disp_img)

    if not input0.IsStreaming() or not input1.IsStreaming() \
        or output.IsStreaming() or frameNo>captureLength:
        input0.Close()
        input1.Close()