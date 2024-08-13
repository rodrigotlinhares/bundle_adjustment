#include "ba.h"

void ba::InitIllum()
{
    std::cout << "Initializing illumination structure..." << std::endl;
    // Allocates control point vectors
    n_ctrl_ptsi = n_ctrl_pts_yi*n_ctrl_pts_xi;
    DefineCtrlPtsIllum();

    // Function values in each control point position
    ctrl_pts_w_i.create(n_ctrl_ptsi, 1, CV_32F);

    // Thin Plate Spline that's going to be approached to the template
    tps.create(size_template, 1, CV_32F);

    // Difference between the TPS and the template
    diff_i.create(size_template, 1, CV_32F);

    // Jacobian
    SD_i.create(size_template, n_ctrl_ptsi, CV_32F);

    // TPS Precomputations
    TPSPrecomputationsIllum();

    ref_intens_value = 100;
}

void ba::ResetIlluminationParam()
{
    ctrl_pts_w_i.setTo(1);
    tps = MKinvi*ctrl_pts_w_i;
}


void ba::IlluminationCompensation(cv::Mat image, cv::Mat mask, cv::Mat& result)
{
    ResetIlluminationParam();

    this->image = image;
    this->mask = mask;

    // eu rodo o gradient descent para equalizacao das imagens por um certo numero de iteracoes. 5
    // eh suficiente na pratica pois geralmente so sao necessarias 2 iteracoes p convergencia
    for(int iters = 0; iters < 5; iters++)
    {
        // Mounting illumination Jacobian
        MountIlluminationJacobian(); // ponto (2)

        // estimating equalization parameters
        UpdateIlluminationParam(); // ponto (3)

        // Computes deformable_mean_image factor
        // ponto (4) corresponde a f(x,i) na notacao no papel
        tps = MKinvi*ctrl_pts_w_i;
    }

    // Computes compensated warped image (faco isso pra levar em consideracao o ultimo update de
    // parametros em updateIlluminationParam())
    ApplyCompensation(result);

    GaussianBlur(result, result, cv::Size(7, 7), 2);

    std::cout << "New deformable_mean_image: " << ctrl_pts_w_i << std::endl;
}

void   ba::ApplyCompensation(cv::Mat& result)
{
    for(int row = 0; row < size_template_y; row++)
        for(int col = 0; col < size_template_x; col++)
        {
            int index = row*size_template_x + col;
//            result.at<uchar>(row, col) = cv::saturate_cast<uchar>((float)image.at<uchar>(row, col) -
//                tps.at<float>(index, 0) + ref_intens_value);
            result.at<uchar>(row, col) = cv::saturate_cast<uchar>(image.at<float>(row, col) -
                tps.at<float>(index, 0) + ref_intens_value);
        }
}

void	ba::MountIlluminationJacobian()
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
//                diff_i.at<float>(index, 0) = tps.at<float>(index,0) - (float)image.at<uchar>(i, j);
                diff_i.at<float>(index, 0) = tps.at<float>(index,0) - image.at<float>(i, j);

                // gradient
                for(int k=0; k<n_ctrl_ptsi; k++)
                {
                    // note que SD_illum tem um numero de colunas igual ao numero de pontos de
                    // controle, e o numero de linhas igual ao número de pixels
                    SD_i.at<float>(index, k) = MKinvi.at<float>(index, k);
                }
            }
            else
            {
                // img difference
                diff_i.at<float>(index, 0) = 0;

                // gradient
                for(int k=0; k<n_ctrl_ptsi; k++)
                    SD_i.at<float>(index, k) = 0;
            }
        }
    }
}

void	ba::TPSPrecomputationsIllum()
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

    for(int i=0;i<size_template_y;i++)
    {
        for(int j=0;j<size_template_x;j++)
        {
            for(int k=0;k<n_ctrl_ptsi;k++)
                Mi.at<float>(j+size_template_x*i, k) = Tps(Norm((float)(j-offx - ctrl_pts_xi[k]),
                                                                (float)(i-offy - ctrl_pts_yi[k])));

            Mi.at<float>(j+size_template_x*i, n_ctrl_ptsi) = 1;
            Mi.at<float>(j+size_template_x*i, n_ctrl_ptsi+1) = (float) j-offx;
            Mi.at<float>(j+size_template_x*i, n_ctrl_ptsi+2) = (float) i-offy;
        }
    }

    MKinvi = Mi*Kinvi;
}

void	ba::DefineCtrlPtsIllum()
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


void    ba::UpdateIlluminationParam()
{
    // ESM update
    cv::Mat delta = ((SD_i.t()*SD_i).inv(CV_SVD)*(SD_i.t()*diff_i));

    // Update parameters
    for(int i=0; i<n_ctrl_ptsi; i++)
        ctrl_pts_w_i.at<float>(i, 0) -= delta.at<float>(i, 0);
}

