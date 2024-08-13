#include "ba.h"

void ba::InitIllum()
{
    std::cout << "Initializing illumination structure..." << std::endl;
    // Allocates control point vectors
    n_ctrl_ptsi = n_ctrl_pts_yi*n_ctrl_pts_xi;
    DefineCtrlPtsIllum();

    // Vector containing tpss values
    ctrl_pts_w_i.first.create(n_ctrl_ptsi, 1, CV_32F);
    ctrl_pts_w_i.second.create(n_ctrl_ptsi, 1, CV_32F);
    tps.first.create(size_template*n_channels, 1, CV_32F);
    tps.second.create(size_template*n_channels, 1, CV_32F);
    diff_i.first.create(size_template, 1, CV_32F);
    diff_i.second.create(size_template, 1, CV_32F);
    SD_i.first.create(size_template, n_ctrl_ptsi, CV_32F);
    SD_i.second.create(size_template, n_ctrl_ptsi, CV_32F);

    // TPS Precomputations
    TPSPrecomputationsIllum();

    ref_intens_value = 100;
}

void ba::ResetIlluminationParam()
{
    ctrl_pts_w_i.first.setTo(1);
    ctrl_pts_w_i.second.setTo(1);
    tps.first = MKinvi*ctrl_pts_w_i.first;
    tps.second = MKinvi*ctrl_pts_w_i.second;
}

void ba::MultichannelIlluminationCompensation(MatPair images, cv::Mat mask, cv::Mat& result)
{
    ResetIlluminationParam();

    this->images = images;
    this->mask = mask;

    // eu rodo o gradient descent para equalizacao das imagens por um certo numero de iteracoes. 5
    // eh suficiente na pratica pois geralmente so sao necessarias 2 iteracoes p convergencia
    for(int iters = 0; iters < 5; iters++)
    {
        // Mounting illumination Jacobian
        MountIlluminationJacobian(); // ponto (2)

        // estimating equalization parameters
        UpdateIlluminationParam(); // ponto (3)

        // Computes tpss factor
        // ponto (4) corresponde a f(x,i) na notacao no papel
        tps.first = MKinvi*ctrl_pts_w_i.first;
        tps.second = MKinvi*ctrl_pts_w_i.second;
    }

    // Computes compensated warped image (faco isso pra levar em consideracao o ultimo update de
    // parametros em updateIlluminationParam())
    ApplyCompensation(result);

    GaussianBlur(result, result, cv::Size(7, 7), 2);

//    std::cout << "New deformable_mean_image: " << ctrl_pts_w_i[0] <<std::endl;
}

void ba::ApplyCompensation(cv::Mat& result)
{
    // Equação (1)
    for(int row = 0; row < size_template_y; row++)
        for(int col = 0; col < size_template_x; col++)
        {
            int index = row*size_template_x + col;
            int value_1 = cv::saturate_cast<uchar>((float)images.first.at<uchar>(row, col) -
                          tps.first.at<float>(index, 0) + ref_intens_value);
            int value_2 = cv::saturate_cast<uchar>((float)images.second.at<uchar>(row, col) -
                          tps.second.at<float>(index, 0) + ref_intens_value);
            result.at<cv::Vec2b>(row, col) = cv::Vec2b(value_1, value_2);
        }
}


void ba::MountIlluminationJacobian()
{
    // Mounting matrix
    for(int i = 0; i < size_template_y; i++)
    {
        for(int j = 0; j < size_template_x; j++)
        {
            int index = j+size_template_x*i;
            if(mask.at<uchar>(i,j) != 0)
            {
                // img difference (vetor que contem as diferencas pra todos os pixels da imagem)
                diff_i.first.at<float>(index, 0) = tps.first.at<float>(index, 0) -
                                                   (float)images.first.at<uchar>(i, j);
                diff_i.second.at<float>(index, 0) = tps.second.at<float>(index, 0) -
                                                    (float)images.second.at<uchar>(i, j);

                // gradient
                for(int k = 0; k < n_ctrl_ptsi; k++)
                {
                    // note que SDs_i tem um numero de colunas igual ao numero de pontos de
                    // controle, e o numero de linhas igual ao número de pixels
                    SD_i.first.at<float>(index, k) = MKinvi.at<float>(index, k);
                    SD_i.second.at<float>(index, k) = MKinvi.at<float>(index, k);
                }
            }
            else
            {
                // img difference
                diff_i.first.at<float>(index, 0) = 0;
                diff_i.second.at<float>(index, 0) = 0;

                // gradient
                for(int k = 0; k < n_ctrl_ptsi; k++)
                {
                    SD_i.first.at<float>(index, k) = 0;
                    SD_i.second.at<float>(index, k) = 0;
                }
            }
        }
    }
}

void ba::TPSPrecomputationsIllum()
{
    // TPS specific
    cv::Mat Mi(size_template, n_ctrl_ptsi+3, CV_32F);
    MKinvi.create(size_template, n_ctrl_ptsi, CV_32F);
    cv::Mat Kinvi(n_ctrl_ptsi+3, n_ctrl_ptsi, CV_32F);

    // TPS Precomputations start here - See TPSPrecomputations for more info
    // Mounting Matrix 'K'
    cv::Mat Ki(n_ctrl_ptsi+3, n_ctrl_ptsi+3, CV_32FC1);

    for(int j=0;j<n_ctrl_ptsi;j++)
    {
        Ki.at<float>(j, n_ctrl_ptsi) = 1;
        Ki.at<float>(j, n_ctrl_ptsi+1) = (float) ctrl_pts_xi[j];
        Ki.at<float>(j, n_ctrl_ptsi+2) = (float) ctrl_pts_yi[j];

        Ki.at<float>(n_ctrl_ptsi, j) = 1;
        Ki.at<float>(n_ctrl_ptsi+1, j) = (float) ctrl_pts_xi[j];
        Ki.at<float>(n_ctrl_ptsi+2, j) = (float) ctrl_pts_yi[j];
    }

    for(int i=0;i<n_ctrl_ptsi;i++)
        for(int j=0;j<n_ctrl_ptsi;j++)
            Ki.at<float>(i, j) = Tps(Norm((float) (ctrl_pts_xi[i]-ctrl_pts_xi[j]),
                                          (float) (ctrl_pts_yi[i]-ctrl_pts_yi[j])));

    for(int i=n_ctrl_ptsi; i<n_ctrl_ptsi+3; i++)
        for(int j=n_ctrl_ptsi; j<n_ctrl_ptsi+3; j++)
            Ki.at<float>(i, j) = 0;

    // Inverting Matrix 'K'
    cv::Mat K2i = Ki.inv(CV_LU);

    // Passing result to Kinv
    for(int i=0;i<n_ctrl_ptsi+3;i++)
        for(int j=0;j<n_ctrl_ptsi;j++)
            Kinvi.at<float>(i, j) = K2i.at<float>(i, j);

    // Creating Matrix 'M'
    int offx = cvCeil((double)size_template_x/2);
    int offy = cvCeil((double)size_template_y/2);

    for(int i = 0; i < size_template_y; i++)
    {
        for(int j = 0; j < size_template_x; j++)
        {
            for(int k = 0; k < n_ctrl_ptsi; k++)
                Mi.at<float>(j+size_template_x*i, k) = Tps(Norm((float)(j-offx - ctrl_pts_xi[k]),
                                                                (float)(i-offy - ctrl_pts_yi[k])));

            Mi.at<float>(j+size_template_x*i, n_ctrl_ptsi) = 1;
            Mi.at<float>(j+size_template_x*i, n_ctrl_ptsi+1) = (float) j-offx;
            Mi.at<float>(j+size_template_x*i, n_ctrl_ptsi+2) = (float) i-offy;
        }
    }

    // Pre-computing M with Kinv
    MKinvi = Mi*Kinvi;
}

void ba::DefineCtrlPtsIllum()
{
    // Initializing control point vector
    ctrl_pts_xi = (int*) malloc(n_ctrl_pts_xi*n_ctrl_pts_yi*sizeof(int));
    ctrl_pts_yi = (int*) malloc(n_ctrl_pts_xi*n_ctrl_pts_yi*sizeof(int));

    int offx = cvCeil((double)size_template_x/2);
    int offy = cvCeil((double)size_template_y/2);

    for(int col = 0; col < n_ctrl_pts_xi; col++)
        for(int row = 0; row < n_ctrl_pts_yi; row++)
        {
            int index = row + col*n_ctrl_pts_yi;
            ctrl_pts_xi[index] = cvRound((col+1)*size_template_x/(n_ctrl_pts_xi+1) - offx);
            ctrl_pts_yi[index] = cvRound((row+1)*size_template_y/(n_ctrl_pts_yi+1) - offy);
        }
}


void ba::UpdateIlluminationParam()
{
    // ESM update
    cv::Mat delta_1 = (SD_i.first.t()*SD_i.first).inv(CV_SVD)*(SD_i.first.t()*diff_i.first);
    cv::Mat delta_2 = (SD_i.second.t()*SD_i.second).inv(CV_SVD)*(SD_i.second.t()*diff_i.second);

    // Update parameters
    for(int i=0; i<n_ctrl_ptsi; i++)
    {
        ctrl_pts_w_i.first.at<float>(i, 0) -= delta_1.at<float>(i, 0);
        ctrl_pts_w_i.second.at<float>(i, 0) -= delta_2.at<float>(i, 0);
    }
}
