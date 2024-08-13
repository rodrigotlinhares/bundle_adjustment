#include "utils.h"

cv::Mat Utils::gradient_image(cv::Mat source)
{                                                                                                    
    cv::Mat result, grad_x, grad_y, abs_grad_x, abs_grad_y;

    Sobel(source, grad_x, CV_32F, 1, 0, 1);
    convertScaleAbs(grad_x, abs_grad_x);

    Sobel(source, grad_y, CV_32F, 0, 1, 1);
    convertScaleAbs(grad_y, abs_grad_y);

    addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, result, CV_32F);
    GaussianBlur(result, result, cv::Size(5, 5), 2);

    normalize(result, result, 0, 255, cv::NORM_MINMAX);

    return result;
} 

cv::Mat Utils::laplacian_of_gaussian(cv::Mat image)
{
    cv::Mat gauss, gray, lap, abs, norm;
    GaussianBlur(image, gauss, cv::Size(3, 3), 0, 0, cv::BORDER_DEFAULT);
    cv::cvtColor(gauss, gray, CV_BGR2GRAY);
    Laplacian(gray, lap, CV_16S, 3, 1, 0, cv::BORDER_DEFAULT);
    convertScaleAbs(lap, abs);
    GaussianBlur(abs, gauss, cv::Size(3, 3), 0, 0, cv::BORDER_DEFAULT);
    normalize(gauss, norm, 0, 255, cv::NORM_MINMAX);
    return norm;
}

void Utils::show_gradient(cv::Mat image, std::string name)
{
    cv::Mat norm;
    normalize(image, norm, 0, 255, cv::NORM_MINMAX, CV_8U);
    cv::imwrite(name, norm);
//    cv::imshow(name, norm);
//    cv::waitKey();
}

int Utils::active_area(cv::Mat mask)
{
    int result = 0;
    for(int row = 0; row < mask.rows; row++)
        for(int col = 0; col < mask.cols; col++)
            if(mask.at<uchar>(row, col) != 0)
                result++;
    return result;
}

float Utils::ncc(cv::Mat image1, cv::Mat image2, cv::Mat mask, int offset_row, int offset_col)
{
    int counter = 0;
    float I1_bar = 0, I2_bar = 0, I1sq = 0, I2sq = 0;
    long error = 0;
    float ncc = 0, stds = 0;

    for(int row = 0; row < image1.rows; row++)
    {
        for(int col = 0; col < image1.cols; col++)
        {
            if(mask.data[row * image1.cols + col] != 0)
            {
                I1_bar += image1.ptr<float>(row)[col];
                I2_bar += image2.ptr<float>(row-offset_row)[col-offset_col];
                counter++;
            }
        }
    }

    if(counter < 1)
    {
        std::cout << "No active pixels to compute NCC!!\n";
        counter = 1;
    }

    I1_bar = I1_bar / counter;
    I2_bar = I2_bar / counter;

    for(int i = 0; i < image2.rows; i++)
    {
        for(int j = 0; j < image2.cols; j++)
        {
            if(mask.data[i * image1.cols + j] != 0)
            {
                I1sq += (image1.ptr<float>(i)[j] - I1_bar) * (image1.ptr<float>(i)[j] - I1_bar);
                I2sq += (image2.ptr<float>(i-offset_row)[j-offset_col] - I2_bar) *
                        (image2.ptr<float>(i-offset_row)[j-offset_col] - I2_bar);
                error += (image1.ptr<float>(i)[j] - I1_bar) *
                         (image2.ptr<float>(i-offset_row)[j-offset_col] - I2_bar);
            }
        }
    }

    stds = std::sqrt(I1sq * I2sq);

    if(stds > 0)
        ncc += (float)error / stds;

    if(ncc < 0)
    {
        std::cout << "Something went wrong when computing the NCC!" << std::endl;
        return 0;
    }
    return ncc;
}

float Utils::ncc_float(cv::Mat image1, cv::Mat image2, cv::Mat mask, int offset_row, int offset_col)
{
    int counter = 0;
    float I1_bar = 0, I2_bar = 0, I1sq = 0, I2sq = 0;
    long error = 0;
    float ncc = 0, stds = 0;

    for(int row = 0; row < image1.rows; row++)
    {
        for(int col = 0; col < image1.cols; col++)
        {
            if(mask.data[row * image1.cols + col] != 0)
            {
                I1_bar += image1.at<float>(row, col);
                I2_bar += image2.at<float>(row-offset_row, col-offset_col);
                counter++;
            }
        }
    }

    if(counter < 1)
    {
        std::cout << "No active pixels to compute NCC!!\n";
        counter = 1;
    }

    I1_bar = I1_bar / counter;
    I2_bar = I2_bar / counter;

    for(int i = 0; i < image2.rows; i++)
    {
        for(int j = 0; j < image2.cols; j++)
        {
            if(mask.data[i * image1.cols + j] != 0)
            {
                I1sq += (image1.at<float>(i, j) - I1_bar) * (image1.at<float>(i, j) - I1_bar);
                I2sq += (image2.at<float>(i-offset_row, j-offset_col) - I2_bar) *
                        (image2.at<float>(i-offset_row, j-offset_col) - I2_bar);
                error += (image1.at<float>(i, j) - I1_bar) *
                         (image2.at<float>(i-offset_row, j-offset_col) - I2_bar);
            }
        }
    }

    stds = std::sqrt(I1sq * I2sq);

    if(stds > 0)
        ncc += (float)error / stds;

    if(ncc < 0)
    {
        std::cout << "Something went wrong when computing the NCC." << std::endl;
        return 0;
    }
    return ncc;
}

float Utils::ncc_uchar(cv::Mat image1, cv::Mat image2, cv::Mat mask, int offset_row, int offset_col)
{
    int counter = 0;
    float I1_bar = 0, I2_bar = 0, I1sq = 0, I2sq = 0;
    long error = 0;
    float ncc = 0, stds = 0;

    for(int row = 0; row < image1.rows; row++)
    {
        for(int col = 0; col < image1.cols; col++)
        {
            if(mask.data[row * image1.cols + col] != 0)
            {
                I1_bar += image1.ptr<uchar>(row)[col];
                I2_bar += image2.ptr<uchar>(row-offset_row)[col-offset_col];
                counter++;
            }
        }
    }

    if(counter < 1)
    {
        std::cout << "No active pixels to compute NCC!!\n";
        counter = 1;
    }

    I1_bar = I1_bar / counter;
    I2_bar = I2_bar / counter;

    for(int i = 0; i < image2.rows; i++)
    {
        for(int j = 0; j < image2.cols; j++)
        {
            if(mask.data[i * image1.cols + j] != 0)
            {
                I1sq += (image1.ptr<uchar>(i)[j] - I1_bar) * (image1.ptr<uchar>(i)[j] - I1_bar);
                I2sq += (image2.ptr<uchar>(i-offset_row)[j-offset_col] - I2_bar) *
                        (image2.ptr<uchar>(i-offset_row)[j-offset_col] - I2_bar);
                error += (image1.ptr<uchar>(i)[j] - I1_bar) *
                         (image2.ptr<uchar>(i-offset_row)[j-offset_col] - I2_bar);
            }
        }
    }

    stds = std::sqrt(I1sq * I2sq);

    if(stds > 0)
        ncc += (float)error / stds;

    if(ncc < 0)
    {
        std::cout << "Something went wrong when computing the NCC!" << std::endl;
        return 0;
    }
    return ncc;
}

float Utils::ncc_color(cv::Mat image1, cv::Mat image2, cv::Mat mask, int offset_row,
                            int offset_col)
{
    int counter = 0;
    int I1_bar[3] = {0}, I2_bar[3] = {0}, I1sq[3] = {0}, I2sq[3] = {0};
    long error[3] = {0};
    float ncc = 0, stds[3] = {0};

    for(int k = 0; k < 3; k++)
    {
        for(int row = 0; row < image1.rows; row++)
        {
            for(int col = 0; col < image1.cols; col++)
            {
                if(mask.data[row * image1.cols + col] != 0)
                {
                    I1_bar[k] += (int)image1.ptr<uchar>(row)[3*col+k];
                    I2_bar[k] += (int)image2.ptr<uchar>(row-offset_row)[3*(col-offset_col)+k];
                    counter++;
                }
            }
        }

        if(counter < 1)
        {
            std::cout << "No active pixels to compute NCC!!\n";
            counter = 1;
        }

        I1_bar[k] = I1_bar[k] / counter;
        I2_bar[k] = I2_bar[k] / counter;
    }

    for(int k = 0; k < 3; k++)
    {
        for(int i = 0; i < image2.rows; i++)
        {
            for(int j = 0; j < image2.cols; j++)
            {
                if(mask.data[i * image1.cols + j] != 0)
                {
                    I1sq[k] += (image1.ptr<uchar>(i)[3*j+k] - I1_bar[k]) *
                               (image1.ptr<uchar>(i)[3*j+k] - I1_bar[k]);
                    I2sq[k] += (image2.ptr<uchar>(i-offset_row)[3*(j-offset_col)+k] - I2_bar[k]) *
                               (image2.ptr<uchar>(i-offset_row)[3*(j-offset_col)+k] - I2_bar[k]);
                    error[k] += (image1.ptr<uchar>(i)[3*j+k] - I1_bar[k]) *
                                (image2.ptr<uchar>(i-offset_row)[3*(j-offset_col)+k] - I2_bar[k]);
                }
            }
        }

        stds[k] = std::sqrt((float)I1sq[k] * (float)I2sq[k]);
    }

    for(int k = 0; k < 3; k++)
        if(stds[k] > 0)
            ncc += (float)error[k] / stds[k];

    if(ncc < 0)
    {
        printf("Something went wrong when computing NCC coef! \n");
        return 0;
    }
    return ncc;
}

long int Utils::ssd_similarity(cv::Mat image1, cv::Mat image2, cv::Mat mask, int offset_row,
                               int offset_col)
{
    long int result = 0;
    for(int row = 0; row < image1.rows; row++)
        for(int col = 0; col < image1.cols; col++)
            if(mask.data[row * image1.cols + col] != 0)
            {
                int sum = image1.at<uchar>(row, col) +
                          image2.at<uchar>(row-offset_row, col-offset_col);
                result += sum * sum;
            }
    return result;
}

float Utils::random(float beginning, float end)
{
    float random = ((float) rand()) / (float) RAND_MAX;
    return random * (end - beginning) + beginning;
}

void Utils::show_vector(cv::Mat vector, int rows, int cols)
{
    cv::Mat out(rows, cols, CV_8U);

    for(int row = 0; row < rows; row++)
        for(int col = 0; col < cols; col++)
            out.at<uchar>(row, col) = vector.at<uchar>(row*cols+col);

    cv::imshow("", out);
    cv::waitKey();
}

void Utils::show_channels(cv::Mat image)
{
    std::vector<cv::Mat> channels;
    split(image, channels);
    cv::imshow("0", channels[0]);
    cv::imshow("1", channels[1]);
    cv::waitKey();
}
