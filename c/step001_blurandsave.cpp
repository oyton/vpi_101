#include <opencv2/core/version.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#if CV_MAJOR_VERSION >= 3
#   include <opencv2/imgcodecs.hpp>
#else
#   include <opencv2/highgui/highgui.hpp>
#endif

#include <iostream>
#include <vpi/OpenCVInterop.hpp>
#include <vpi/Image.h>
#include <vpi/Stream.h>

#include <vpi/algo/BoxFilter.h>
#include <vpi/algo/ConvertImageFormat.h>

int main(int argc, char *argv[])
{
    if (argc != 2)
    {
        std::cerr << "Must pass an input image to be blurred and saved" << std::endl;
        return 1;
    }

    cv::Mat cvImage = cv::imread(argv[1]);
    if (cvImage.data == NULL)
    {
        std::cerr << "Can't open input image" << std::endl;
    }

    VPIStream stream;
    vpiStreamCreate(0, &stream); //0 allows algorithms submitted to it to run in any available backend.

    VPIImage image;
    vpiImageCreateOpenCVMatWrapper(cvImage, 0, &image);

    VPIImage imageGray;
    vpiImageCreate(cvImage.cols, cvImage.rows, VPI_IMAGE_FORMAT_U8, 0, &imageGray);

    VPIImage blurred;
    vpiImageCreate(cvImage.cols, cvImage.rows, VPI_IMAGE_FORMAT_U8, 0, &blurred);

    vpiSubmitConvertImageFormat(stream, VPI_BACKEND_CUDA, image, imageGray, NULL);
    vpiSubmitBoxFilter(stream, VPI_BACKEND_CUDA, imageGray, blurred, 5, 5, VPI_BORDER_ZERO);

    vpiStreamSync(stream);

    VPIImageData outData;
    vpiImageLock(blurred, VPI_LOCK_READ, &outData);

    cv::Mat cvOut;
    vpiImageDataExportOpenCVMat(outData, &cvOut);
    imwrite("step001_blurandsave.png", cvOut);

    vpiImageUnlock(blurred);

    vpiStreamDestroy(stream);
    vpiImageDestroy(image);
    vpiImageDestroy(imageGray);
    vpiImageDestroy(blurred);

    return 0;
    
}