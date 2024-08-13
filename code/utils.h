#ifndef UTILS_H_
#define UTILS_H_

#include <cv.h>
#include <highgui.h>

class Utils
{
public:
    static cv::Mat gradient_image(cv::Mat source);

    static cv::Mat laplacian_of_gaussian(cv::Mat image);

    static void show_gradient(cv::Mat image, std::string name);

    static int active_area(cv::Mat mask);

    static float ncc(cv::Mat image1, cv::Mat image2, cv::Mat mask, int offset_row, int offset_col);

    static float ncc_float(cv::Mat image1, cv::Mat image2, cv::Mat mask, int offset_row,
                           int offset_col);

    static float ncc_uchar(cv::Mat image1, cv::Mat image2, cv::Mat mask, int offset_row,
                           int offset_col);

    static float ncc_color(cv::Mat image1, cv::Mat image2, cv::Mat mask, int offset_row,
                                int offset_col);
    static long int ssd_similarity(cv::Mat image1, cv::Mat image2, cv::Mat mask, int offset_row,
                                   int offset_col);
    static float random(float begin, float end);

    static void show_vector(cv::Mat vector, int rows, int cols);

    static void show_channels(cv::Mat image);
};

#endif
