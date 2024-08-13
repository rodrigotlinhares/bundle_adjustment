#include "ba.h"

ba::~ba()
{
    ncc_stats.close();
    ssd_stats.close();
}

void ba::InitializeMosaic(int grid_x,
                          int grid_y,
                          int offset_templates,
                          int size_template_x,
                          int size_template_y,
                          int n_active_pixels,
                          int n_ctrl_pts_x,
                          int n_ctrl_pts_y,
                          int n_ctrl_pts_xi,
                          int n_ctrl_pts_yi,
                          int n_max_iters,
                          float epsilon,
                          int interp,
                          cv::Mat Visibility_map,
                          std::vector<cv::Point> anchors)
{
    // Setup
    this->grid_x = grid_x;
    this->grid_y = grid_y;
    this->offset_templates = offset_templates;
    this->size_template_x = size_template_x;
    this->size_template_y = size_template_y;
    this->n_active_pixels = n_active_pixels;
    this->n_ctrl_pts_x = n_ctrl_pts_x;
    this->n_ctrl_pts_y = n_ctrl_pts_y;
    this->n_ctrl_pts_xi = n_ctrl_pts_xi;
    this->n_ctrl_pts_yi = n_ctrl_pts_yi;
    this->lambda = lambda;
    this->epsilon = epsilon;
    this->n_max_iters = n_max_iters;
    this->interp = interp;
    this->Visibility_map = Visibility_map;
    this->anchor_positions = anchors;
    this->using_masks = true;
    this->using_colors = true;
    this->size_template = size_template_x*size_template_y;
    this->n_channels = 2;

    // Open output for stats
    ncc_stats.open("../storage/ncc.txt");
    ssd_stats.open("../storage/ssd.txt");

    // Process visibility_map
    ProcessVisibilityMap();

    // Process illumination parameters
    InitIllum();

    // Reference pixel list when n_active_pixels = size_template_x*size_template_y
    std_pixel_list = (int*)malloc(size_template_x*size_template_y*sizeof(int));
    for(int i=0;i<size_template_x*size_template_y;i++)
        std_pixel_list[i] = i;

    active_pixels_r = std_pixel_list;
    active_pixels_g = std_pixel_list;
    active_pixels_b = std_pixel_list;

    // Allocates control point vectors
    n_ctrl_pts = n_ctrl_pts_y*n_ctrl_pts_x;
    DefineCtrlPts();

    // Now initialize ctrl_pts_w
    ctrl_pts_x_w.resize(n_elements);
    ctrl_pts_y_w.resize(n_elements);

    for(int i=0; i<n_elements; i++)
    {
        ctrl_pts_x_w[i].create(n_ctrl_pts, 1, CV_32F);
        ctrl_pts_y_w[i].create(n_ctrl_pts, 1, CV_32F);
    }

    for(int j=0; j<n_elements; j++)
    {
        for(int i=0; i<n_ctrl_pts; i++)
        {
            ctrl_pts_x_w[j].at<float>(i,0) = ctrl_pts_x[i]+ cvCeil(size_template_x/2);
            ctrl_pts_y_w[j].at<float>(i,0) = ctrl_pts_y[i]+ cvCeil(size_template_y/2);
        }
    }

    // Initialize compensated image vectors
    ICurComp.resize(n_elements);
    for(int i=0; i < n_elements; i++)
        ICurComp[i].create(size_template_y, size_template_x, CV_8UC2);

	// TPS specific
	M.create(size_template_x*size_template_y, n_ctrl_pts+3, CV_32F);
    MKinv.create(size_template_x*size_template_y, n_ctrl_pts, CV_32F);
    MKinvT.create(n_ctrl_pts, size_template_x*size_template_y, CV_32F);
    Kinv.create(n_ctrl_pts+3, n_ctrl_pts, CV_32F);
	Ks.create(2*n_ctrl_pts, 2*n_ctrl_pts, CV_32F);
	Ksw.create(2*n_ctrl_pts, 1, CV_32F);

    // TPS Precomputations
    TPSPrecomputations();

    // The dummy mapping matrices' shape is different in the TPS code
    dummy_mapx.resize(NUM_THREADS);
    dummy_mapy.resize(NUM_THREADS);
    for(int i = 0; i < NUM_THREADS; i++)
    {
        dummy_mapx[i].create(size_template, 1, CV_32F);
        dummy_mapy[i].create(size_template, 1, CV_32F);
    }

    // The rest is pretty much standard
    SDx.resize(n_elements);
    gradx.resize(n_elements);
    grady.resize(n_elements);

    current_warp.resize(n_elements);
    current_mask = new cv::Mat [n_elements];
    current_warp_color = new cv::Mat [n_elements];

    for(int i=0; i<n_elements; i++)
    {
        // transposto com relacao a orientacao que eu assumi no papel
        SDx[i].create(2*n_ctrl_pts, n_active_pixels*2, CV_32F);

        gradx[i].create(size_template_x, size_template_y, CV_32FC2);
        grady[i].create(size_template_x, size_template_y, CV_32FC2);

        current_warp[i].create(size_template_y, size_template_x, CV_8UC2);
        current_mask[i].create(size_template_y, size_template_x, CV_8UC1);
        current_warp_color[i].create(size_template_y, size_template_x, CV_8UC3);
    }

    dif.resize(n_combinations);
    SDw.resize(n_combinations);
    combination_masks.resize(n_combinations);

    for(int i=0; i<n_combinations; i++)
    {
        dif[i].create(n_active_pixels*2, 1, CV_32F);
        SDw[i].create(2*n_ctrl_pts, n_active_pixels*2, CV_32F);
        combination_masks[i].create(n_active_pixels, 1, CV_8U);
    }

    // Gradient
    gradient.create(2*n_ctrl_pts*(n_elements-n_anchors), 1, CV_32F);

    // Hessian matrices
    hessian.create(2*n_ctrl_pts*(n_elements-n_anchors),2*n_ctrl_pts*(n_elements-n_anchors), CV_32F);
    dummy_hessian.create(2*n_ctrl_pts, 2*n_ctrl_pts, CV_32FC1);
    hess_precomp.resize(n_elements);

    for(int i=0; i<n_elements; i++)
        hess_precomp[i].push_back(dummy_hessian.clone());

    // Misc
    Erosion_mask = getStructuringElement( cv::MORPH_ELLIPSE, cv::Size(3, 3), cv::Point(1,1));

    locks = new omp_lock_t [n_elements];
    for(int i = 0; i < n_elements; i++)
        omp_init_lock(&locks[i]);

}



// Mosaicking

void ba::Process(cv::Mat ICur[], cv::Mat MaskICur[], cv::Mat ICurColor[])
{
    // Compensating illumination on green and red channels in all templates
    for(int i = 0; i < n_elements; i++)
    {
        std::vector<cv::Mat> channels;
        split(ICurColor[i], channels);
        MultichannelIlluminationCompensation(std::make_pair(channels[1], channels[2]), MaskICur[i],
                                             ICurComp[i]);

        merge(channels, ICurComp[i]);
    }

    // Aqui eu calculo os T' pela primeira vez para o elemento ancora, ja que ele eh igual a
    // T (T'=T)
    // eu poderia ter copiado T para T' mas eu tive muita preguica
    // Note que T' eh chamado de 'current_warp'
    // As componentes x e y ds pontos de controle 'p' sao chamados de ctrl_pts_x_w e ctrl_pts_y_w,
    // respectivamente
    // Ponto 5 - eu uso os vetores acima para calcular as componentes x e y dos pixels 'warpados',
    // para cada template T'
//    WarpTPS(&ICurComp[n_fixed_element], &MaskICur[n_fixed_element], &current_warp[n_fixed_element],
//            &current_mask[n_fixed_element], &ctrl_pts_x_w[n_fixed_element],
//            &ctrl_pts_y_w[n_fixed_element]);
//    WarpTPSColor(&ICurColor[n_fixed_element], &current_warp_color[n_fixed_element]);
    for(int i = 0; i < anchors.size(); i++)
    {
        current_warp[anchors[i]] = ICurComp[anchors[i]];
        current_mask[anchors[i]] = MaskICur[anchors[i]];
        current_warp_color[anchors[i]] = ICurColor[anchors[i]];
    }

    // Minimization loop
    for(int iterations = 0; iterations < n_max_iters; iterations++)
    {
        // Start clock
        double cs = cvGetTickCount();

        // Warping images and computing gradient
        // Aqui eu calculo os T' pela primeira vez, usando os parametros p iniciais (exceto para o
        // elemento ancora), assim como as mascaras M'
        #pragma omp parallel for
        for(int index = 0; index < n_elements; index++) 
            if(!is_anchor(index))
            {
                WarpTPS(ICurComp[index], &MaskICur[index], current_warp[index],
                        &current_mask[index], ctrl_pts_x_w[index], ctrl_pts_y_w[index]);
                WarpTPSColor(&ICurColor[index], &current_warp_color[index]);
            }

        // Ponto (6), para o calculo posterior da jacobiana, eh necessario o calculo do gradiente
        // da imagem T'
        #pragma omp parallel for
        for(int index = 0; index < n_elements; index++)
            if(!is_anchor(index))
                WarpGrad(current_warp[index], gradx[index], grady[index]);

        // Ponto (7) Building individual jacobians
        #pragma omp parallel for
        for(int index = 0; index < n_elements; index++)
            if(!is_anchor(index))
                BuildIndividualMosaicJacobian(index);

        // Criar aqui a funcao ComputeImageWeights()
        // voce vai atualizar w[block]
        ComputeCombinationMasks();

        for(int block = 0; block < n_combinations; block++)
        {
            ncc_stats << Utils::ncc_similarity(current_warp_color[combinations[block][0]],
                                               current_warp_color[combinations[block][1]],
                                               combination_masks[block],
                                               combinations[block][2],
                                               combinations[block][3]) << " ";
            ssd_stats << Utils::ssd_similarity(current_warp_color[combinations[block][0]],
                                               current_warp_color[combinations[block][1]],
                                               combination_masks[block],
                                               combinations[block][2],
                                               combinations[block][3]) << " ";
        }
        ncc_stats << "\n";
        ssd_stats << "\n";

        // Para toda entrada na minha lista de correspondencias 'combinations', eu devo calcular a
        // diferenca de intensidade entre as imagens
        // No exemplo do papel, eu devo calcular Delta I0 e Delta I1, correspondentes as
        // correpondencias 0-1 e 1-2
        // Logo, devo calcular a diferenca de intensidade entre T0' e T1' e T1' e T2'
        // NO ENTANTO, as correspondencias tem um offset entre si, que eh pre determinado nas
        // posicoes 3-4 do vetor combinations
        // Logo, a funcao abaixo calcula os Delta I de maneira apropriada

        // Computing image difference
        ComputeImageDifference();

        std::cout << "Time it takes to compute all warps, jacobians and image differences (ms): "
                  << (cvGetTickCount()-cs)/(1000*cvGetTickFrequency()) << std::endl;

        // nessa funcao, usando as jacobianas de cada template T' previamente calculadas no
        // ponto (7), eu devo mapear a derivada (jacobiana) de cada template com offset da minha
        // lista de correspondencias
        // e colocar o resultado em uma matriz chamada 'SDw'
        // note que eu nao preciso calcular a jacobiana com offset para o template ancora
        // no exemplo do papel, eu nao tive que calcular a jacobiana com offset para o primeiro
        // termo de alpha, ja que T1 era o template ancora
        BuildOffsetJacobians();

        std::cout << "Time it takes to compute all above + offsets (ms): "
                  << (cvGetTickCount()-cs)/(1000*cvGetTickFrequency()) << std::endl;

        // O gradiente eh a multiplicacao das jacobianas com as diferencas para cada combinacao na
        // minha lista de combinacoes

        // ponto (10)
        ComputeGradient();

        std::cout << "Time it takes to compute all above + gradient (ms): "
                  << (cvGetTickCount()-cs)/(1000*cvGetTickFrequency()) << std::endl;

        // Fazer hessianas
        // esse calculo ainda nao esta bom, precisamos de atualizar essa funcao aqui e seu mestrado
        // estara pronto
        ComputeBigHessian();

        std::cout << "Time it takes to compute all above + hessian (ms): "
                  << (cvGetTickCount()-cs)/(1000*cvGetTickFrequency()) << std::endl;

        if(UpdateParams())
        {
            std::cout << "Numero de iteracoes: " << iterations << std::endl;
            break;
        }

        std::cout << "Time it took to compute everything (ms): "
                  << (cvGetTickCount()-cs)/(1000*cvGetTickFrequency()) << std::endl;

        // Display
        std::cout << ">> End of iteration: " << iterations << std::endl << std::endl;

//        // Show all images in a grid (only works for Lena).
//        std::stringstream ss;
//        int index = 0, size = 100;
//        cv::Mat black = cv::Mat::zeros(size, size, CV_8U);
//        for(int col = 0; col < grid_x; col++)
//        {
//            cv::Mat small;
//            for(int row = 0; row < grid_y; row++)
//            {
//                ss << "(" << row << "," << col << ")";
//                if(Visibility_map.at<uchar>(row, col) != 0)
//                {
//                    resize(current_warp_color[index], small, cv::Size(size, size));
//                    cv::imshow(ss.str(), small);
//                    index++;
//                }
//                else
//                    cv::imshow(ss.str(), black);
//                cv::moveWindow(ss.str(), 1921+col*(size+4), 50+row*(size+30));
//                ss.str("");
//            }
//        }

        for(int index=0; index<n_elements; index++)
        {
            char text[30];
            sprintf(text, "t'%d", index);
            cv::imshow(text, current_warp_color[index]);

//                sprintf(text, "m'%d", index);
//                cv::imshow(text, current_mask[index]);
        }

        cv::waitKey(10);
    }
}

// Misc
int    ba::UpdateParams()
{
    // ponto 14
    cv::Mat delta = hessian.inv()*gradient;

    // Update
    int check = 0, offset = 0;
    for(int index=0; index<n_elements; index++)
    {
        if(is_anchor(index))
            offset++;
        else
        {
            float sum = 0;

            for(int i=0; i<n_ctrl_pts; i++)
            {
                ctrl_pts_x_w[index].at<float>(i,0) -= delta.at<float>((index-offset)*2*n_ctrl_pts + i, 0);
                sum += fabs(delta.at<float>((index-offset)*2*n_ctrl_pts + i, 0));
                ctrl_pts_y_w[index].at<float>(i,0) -= delta.at<float>((index-offset)*2*n_ctrl_pts + i + n_ctrl_pts, 0);
                sum += fabs(delta.at<float>((index-offset)*2*n_ctrl_pts +i + n_ctrl_pts, 0));
            }

            if(sum < epsilon) check++;
        }
    }

    std::cout << check << " converged out of " << n_elements - anchors.size() <<  std::endl;

    return check == n_elements - anchors.size();
}

cv::Mat ba::ComputeHessian(cv::Mat* Jacobian1, cv::Mat *Jacobian2, cv::Mat mask)
{
    cv::Mat result = cv::Mat::zeros(2*n_ctrl_pts, 2*n_ctrl_pts, CV_32F);

    // Computing dot products
    #pragma omp parallel for
    for(int i = 0; i < 2*n_ctrl_pts; i++)
        for(int j = i; j < 2*n_ctrl_pts; j++)
            for(int k = 0; k < Jacobian1->cols; k++)
                if(mask.data[k] != 0)
                    result.at<float>(i,j) += Jacobian1->at<float>(i,k) * Jacobian2->at<float>(j,k);

    // Reflecting Hessians
    #pragma omp parallel for
    for(int i = 0; i < 2*n_ctrl_pts; i++)
        for(int j = i; j < 2*n_ctrl_pts; j++)
            result.at<float>(j,i) = result.at<float>(i,j);

    return result;
}


void    ba::ComputeBigHessian()
{
    // Clean hessian
    hessian.setTo(0);

    // nesse primeiro momento, eu calculo as componentes parcial alpha / (parcial p0, p0) , parcial alpha / (parcial p1, p1) ...
    // que sao as diagonais da matriz H indicada no ponto (23)
    // O problema aqui ´e que eu nao levo em consideracao os pesos w de cada combinacao

    // for every pair of correspondences
    for(int block=0; block<n_combinations; block++)
    {
        int offset = 2*n_ctrl_pts;

        int first = combinations[block][0];
        int second = combinations[block][1];

        // Fixing index because anchor template is not in big hessian
        int hessian_index_1 = first - anchors_before(first);
        int hessian_index_2 = second - anchors_before(second);

        // aqui eu devo calcular a hessiana correspondente ao primeiro elemento da combinacao 'block'
        // como faço isso? H = J^t J, onde cada J leva em consideracao os pesos w[block] (veja folha 16)
        if(!is_anchor(first))
        {
            cv::Mat little_hessian = ComputeHessian(&SDx[first], &SDx[first], combination_masks[block]);
            hessian(cv::Rect(hessian_index_1*offset, hessian_index_1*offset, offset, offset)) += little_hessian; // Adding all components of dAlpha / dpi Ponto (24)
        }

        // O mesmo se aplica pro segundo elemento da combinaçao. No entanto, aqui eu devo usar a
        // Jacobiana warpada correspondente
        // note que tamb´em devo levar em consideracao os pesos w[block] apropriados

        // TODO RODRIGO nesse final de semana: voce vai ter que modificar a funcao computeHessian
        // para levar em consideracao os pesos w
        // no codigo atualmente nao existe essa estrutura de pesos w. voce vai ter que criar ela,
        // calcular, para cada combinacao, a cada iteracao, os
        // pesos w[block]
        if(!is_anchor(second))
        {
            cv::Mat little_hessian = ComputeHessian(&SDw[block], &SDw[block], combination_masks[block]);
            hessian(cv::Rect(hessian_index_2*offset, hessian_index_2*offset, offset, offset)) += little_hessian; // ponto (24)
        }

        // Aqui eu faço a hessiana usando as jacobianas do primeiro template (SDx) e do segundo
        // template warpado (SDw)
        // note que eu tambem nao levo em conta os pesos w[block]
        if(!is_anchor(first) && !is_anchor(second))
        {
            cv::Mat little_hessian = ComputeHessian(&SDx[first], &SDw[block], combination_masks[block]);
            hessian(cv::Rect(hessian_index_1*offset, hessian_index_2*offset, offset, offset)) -= little_hessian; // ponto (24)
        }
    }

    // Reflecting Hessian
    #pragma omp parallel for
    for(int i=0;i<hessian.rows;i++)
        for(int j=i;j<hessian.cols;j++)
            hessian.at<float>(j,i) = hessian.at<float>(i,j);
}

void    ba::BuildOffsetJacobians()
{
    // For every combination
    #pragma omp parallel for
    for(int block=0; block<n_combinations; block++)
    {
        // I take the offset from the second template pos_1, (que esta em offset_x e offset_y)
        int pos_1 = combinations[block][1];
        int offset_x = combinations[block][2];
        int offset_y = combinations[block][3];

        // If the second template is the fixed element (ancora), I do nothing
        if(!is_anchor(pos_1))
        {
            // For all pixels on image, I compute operation (9)
            for(int k=0; k<2*n_ctrl_pts; k++)
                for(int index=0; index<n_active_pixels; index++)
                {
                    // 'index' corresponde a indices (j + i*(numero de pxl por coluna) de pixel na
                    // imagem original T'
                    int i = cvFloor((float)index/size_template_x);
                    int j = index - i*size_template_x;

                    // enquanto que 'index2' percorre os pixels da imagem com offset
                    // de cuja imagem eu desejo calcular a jacobiana
                    int index2 = (i-offset_y)*size_template_x+(j-offset_x);

                    // bem aqui eu pego as entradas de SDx, que corresponde a Jacobiana calculada
                    // para as imagens originais T' e copio elas em SDw, que contem a jacobiana para
                    // T' offsetado
                    // I copy the offset pixels to their new positions
                    if(combination_masks[block].data[index] != 0)
                    {
                        SDw[block].at<float>(k, index) = SDx[pos_1].at<float>(k, index2);
                        SDw[block].at<float>(k, index+n_active_pixels) = SDx[pos_1].at<float>(k, index2+n_active_pixels);
                    }
                    else
                    {
                        SDw[block].at<float>(k, index) = 0;
                        SDw[block].at<float>(k, index+n_active_pixels) = 0;
                    }
                }
        }
    }
}


void	ba::ProcessVisibilityMap()
{
    std::cout << "Processing visibility map..." << std::endl;
    n_elements = 0;

    cv::Mat Element_number(Visibility_map.rows, Visibility_map.cols, CV_32SC1);
    Element_number.setTo(0);

    // Primeiro contamos o número de elementos
    for(int j=0; j<grid_x; j++)
        for(int i=0; i<grid_y; i++)
            if(Visibility_map.at<uchar>(i,j) != 0)
            {
                Element_number.at<int>(i,j) = n_elements;
                for(int k = 0; k < anchor_positions.size(); k++)
                    if(anchor_positions[k] == cv::Point(i, j))
                        anchors.push_back(n_elements);
                n_elements++;
            }

    n_anchors = anchors.size();

    // Alocação do vetor de combinações
    combinations = (int**) malloc(10000*sizeof(int*));

    for(int i=0; i<10000; i++)
        combinations[i] = (int*) malloc(4*sizeof(int));

    n_combinations = 0;

    // Max offsets that guarantee a 50% overlap
//    int max_offset_x = size_template_x / (2 * offset_templates);
//    int max_offset_y = size_template_y / (2 * offset_templates);
//    int max_offset_diagonal = size_template_x + size_template_y +
//        sqrt(size_template_x*size_template_x + size_template_y*size_template_y) / 2*offset_templates;

    //TODO fix
    int max_offset_x = 2;
    int max_offset_y = 2;
    int max_offset_diagonal = 1;

    for(int j=0; j<grid_x; j++)
        for(int i=0; i<grid_y; i++)
        {
            if(Visibility_map.at<uchar>(i,j) != 0)
            {
                // Closest neighbor below
                for(int offset = 1; i + offset < grid_y && offset <= max_offset_y; offset++)
                {
                    if(Visibility_map.at<uchar>(i + offset, j) != 0)
                    {
                        combinations[n_combinations][0] = Element_number.at<int>(i, j);
                        combinations[n_combinations][1] = Element_number.at<int>(i + offset, j);
                        combinations[n_combinations][2] = 0;
                        combinations[n_combinations][3] = offset * offset_templates;
                        n_combinations++;
                        break;
                    }
                }

                // Closest neighbor to the right
                for(int offset = 1; j + offset < grid_x && offset <= max_offset_x; offset++)
                {
                    if(Visibility_map.at<uchar>(i, j + offset) != 0)
                    {
                        combinations[n_combinations][0] = Element_number.at<int>(i, j);
                        combinations[n_combinations][1] = Element_number.at<int>(i, j + offset);
                        combinations[n_combinations][2] = offset * offset_templates;
                        combinations[n_combinations][3] = 0;
                        n_combinations++;
                        break;
                    }
                }

                // Closest lower-left neighbor
                for(int offset = 1; i + offset < grid_y && j - offset >= 0 &&
                    offset <= max_offset_diagonal; offset++)
                {
                    if(Visibility_map.at<uchar>(i + offset, j - offset) != 0)
                    {
                        combinations[n_combinations][0] = Element_number.at<int>(i, j);
                        combinations[n_combinations][1] = Element_number.at<int>(i + offset,
                                                                                 j - offset);
                        combinations[n_combinations][2] = -offset * offset_templates;
                        combinations[n_combinations][3] = offset * offset_templates;
                        n_combinations++;
                        break;
                    }
                }

                // Closest lower-right neighbor
                for(int offset = 1; i + offset < grid_y && j + offset < grid_x &&
                    offset <= max_offset_diagonal; offset++)
                {
                    if(Visibility_map.at<uchar>(i + offset, j + offset) != 0)
                    {
                        combinations[n_combinations][0] = Element_number.at<int>(i, j);
                        combinations[n_combinations][1] = Element_number.at<int>(i + offset,
                                                                                 j + offset);
                        combinations[n_combinations][2] = offset * offset_templates;
                        combinations[n_combinations][3] = offset * offset_templates;
                        n_combinations++;
                        break;
                    }
                }
            }
        }

    std::cout << " Here are all possible combinations:" << std::endl;

    for(int i=0; i<n_combinations; i++)
    {
        std::cout << combinations[i][0] << " " << combinations[i][1] << " " << combinations[i][2]
                  << " " << combinations[i][3] << std::endl;
        ncc_stats << combinations[i][0] << " ";
        ssd_stats << combinations[i][0] << " ";
    }
    ncc_stats << "\n";
    ssd_stats << "\n";

    for(int i=0; i<n_combinations; i++)
    {
        ncc_stats << combinations[i][1] << " ";
        ssd_stats << combinations[i][1] << " ";
    }
    ncc_stats << "\n";
    ssd_stats << "\n";
}


// *** nesta funcao eu estou adicionando cada diferenca multiplicada pelas jacobianas apropriadas
// (SDx e SDw, para o primeiro e segundo elementos da combinacao)
// nas colunas correspondntes de G (ponto 12, atras da pagina 6)
void    ba::ComputeGradient()
{
    // esse vetor corresponde ao vetor G do ponto (11) e ponto (10)
    // Resetting gradient vector
    gradient.setTo(0);

    // no loop abaixo, eu vou percorrer cada combinacao.
    // para o primeiro e segundo elemento da combinacao, eu vou calcular o vetor de gradiente
    // 'combination_gradient' para cada
    // elemento separadamente e adiciona-lo na posicao apropriada em G (vetor 'gradient')
    #pragma omp parallel for
    for(int block=0; block<n_combinations; block++)
    {
        int first = combinations[block][0];
        int second = combinations[block][1];

        int index_column1 = first - anchors_before(first);
        int index_column2 = second - anchors_before(second);
        int second_half_offset = 2*n_ctrl_pts*(n_elements-n_anchors);

        // For every correspondence, we compute the product between Jacobians and image difference
        if(!is_anchor(first))
        {
            // note que SDx foi montado transposto de proposito, para essa operacao aqui
            // note tambem coitado, que voce pode melhorar essa operacao divindo ainda mais
            // os calculos entre os processadores da sua maquina, usando openmp
            cv::Mat combination_gradient = SDx[first]*dif[block];

            // Em seguida eu incremento a posicao no vetor 'gradient'
            omp_set_lock(&locks[first]);
            gradient(cv::Rect(0,index_column1*2*n_ctrl_pts, 1, 2*n_ctrl_pts))
                += combination_gradient;
            omp_unset_lock(&locks[first]);

//            omp_set_lock(&locks[first]);
//            gradient(cv::Rect(0, index_column1*2*n_ctrl_pts+second_half_offset, 1, 2*n_ctrl_pts))
//                += combination_gradient;
//            omp_unset_lock(&locks[first]);
        }

        if(!is_anchor(second))
        {
            // note que aqui eu uso a jacobiana warpada do template T'
            // e note tambem que a jacobiana warpada eh diferente para cada combinacao, ja que o
            // offset muda para cada comb.
            cv::Mat combination_gradient = -SDw[block]*dif[block];  // ponto (13)

            // Em seguida eu incremento a posicao no vetor 'gradient'
            omp_set_lock(&locks[second]);
            gradient(cv::Rect(0,index_column2*2*n_ctrl_pts,1,2*n_ctrl_pts))
                += combination_gradient;
            omp_unset_lock(&locks[second]);

//            omp_set_lock(&locks[first]);
//            gradient(cv::Rect(0, index_column2*2*n_ctrl_pts+second_half_offset, 1, 2*n_ctrl_pts))
//                += combination_gradient;
//            omp_unset_lock(&locks[first]);
        }
    }
}

void ba::ComputeCombinationMasks()
{
    #pragma omp parallel for
    for(int pair = 0; pair < n_combinations; pair++)
    {
        cv::Mat mask_0 = current_mask[combinations[pair][0]];
        cv::Mat mask_1 = current_mask[combinations[pair][1]];
        int col_offset = combinations[pair][2];
        int row_offset = combinations[pair][3];

        for(int row = 0; row < size_template_y; row++)
        {
            for(int col = 0; col < size_template_x; col++)
            {
                int index = row * size_template_x + col;

                if(row >= row_offset &&
                   col >= col_offset &&
                   (col - col_offset) < size_template_x &&
                   (row - row_offset) < size_template_y)
                {
                    combination_masks[pair].at<uchar>(index, 0) = mask_0.at<uchar>(row, col) &&
                        mask_1.at<uchar>(row-row_offset, col-col_offset);
                }
            }
        }
    }
}

void ba::BuildIndividualMosaicJacobian(int element)
{
	// Bulding individual Jacobians
    for(int k=0; k<n_ctrl_pts; k++)
        for(int index=0; index<n_active_pixels; index++)
        {
            int index2 = index+n_active_pixels;
            // Active pixel positions
            int i = cvFloor((float)index/size_template_x);
            int j = index - i*size_template_x;

            // If it's an active pixel
            // ponto (21)
            // isso aqui esta realmente errado/incompleto
            {
                // Build Jacobian image 1
                // 'x' derivative
                cv::Vec2f grad_x_value = gradx[element].at<cv::Vec2f>(i, j);
                SDx[element].at<float>(k, index) = MKinvT.at<float>(k,index) * grad_x_value[0];
                SDx[element].at<float>(k, index2) = MKinvT.at<float>(k,index) * grad_x_value[1];

                // 'y' derivative
                cv::Vec2f grad_y_value = grady[element].at<cv::Vec2f>(i, j);
                SDx[element].at<float>(k+n_ctrl_pts, index) = MKinvT.at<float>(k,index) *
                                                              grad_y_value[0];
                SDx[element].at<float>(k+n_ctrl_pts, index2) = MKinvT.at<float>(k,index) *
                                                               grad_y_value[1];
            }
        }
}

void ba::ComputeImageDifference()
{
    // como eu faco isso de maneira automatizada, eu processo cada entrada da minha lista de
    // correspondencias, descubro qual sao os T' pareados e leio o offset entre eles. Os indices do
    // par de templates eh determinado em pos_0 e pos_1, ok?
    // Em seguida, o offset de pos_1 eh determinado como offset_x e y

//    #pragma omp parallel for
    for(int block=0; block<n_combinations; block++)
    {
        int pos_0 = combinations[block][0];
        int pos_1 = combinations[block][1];
        int offset_x = combinations[block][2];
        int offset_y = combinations[block][3];

        // Computing pixel difference
        // lembre-se que n_active_pixels quer dizer simplesmente numero de pixels na imagem T'
        for(int index = 0; index < n_active_pixels; index++)
        {
            int index2 = index + n_active_pixels;
            int i = cvFloor((float)index/size_template_x);
            int j = index - i*size_template_x;

            // ponto (22) - TODO - atualmente, eu so desativo os pixels qu enao estao na
            // interseccao. Aqui vou ter que desativar os pixels que estao desativados em
            // current_mask[pos_0] e
            // current_mask[pos_1] (com offset) lembre-se rodrigo, com offset apropriado, por
            // favor , nao crie mais bugs que o necessario, claro que faz a diferenca
            if(combination_masks[block].data[index] != 0)
            {
                cv::Vec2b value_0 = current_warp[pos_0].at<cv::Vec2b>(i, j);
                cv::Vec2b value_1 = current_warp[pos_1].at<cv::Vec2b>(i-offset_y, j-offset_x);
                // ponto (7)
                dif[block].at<float>(index, 0) = (float)value_0[0] - (float)value_1[0];
                dif[block].at<float>(index2, 0) = (float)value_0[1] - (float)value_1[1];
            }
            else
            {
                dif[block].at<float>(index, 0) = 0;
                dif[block].at<float>(index2, 0) = 0;
            }
        }
    }
}

void ba::WarpTPS(cv::Mat ICur, cv::Mat *Mask_roi, cv::Mat& current_warp, cv::Mat *current_mask,
                 cv::Mat ctrl_pts_x_w, cv::Mat ctrl_pts_y_w)
{
    int proc_id = omp_get_thread_num();
	// Mapping pixel positions
	dummy_mapx[proc_id] = MKinv*ctrl_pts_x_w;
	dummy_mapy[proc_id] = MKinv*ctrl_pts_y_w;

    // Remapping (aqui eu calculo t')
	cv::remap(ICur, current_warp, dummy_mapx[proc_id].reshape(1, size_template_y),
              dummy_mapy[proc_id].reshape(1, size_template_y), cv::INTER_CUBIC, 0,
              cvScalar(0));

    // e aqui eu calculo (m'-> 'current_mask') para todos os templates do meu mosaico
    // Ponto (20)
	if(using_masks)
	{
        // nearest neighbors para velocidade
        cv::remap(*Mask_roi, *current_mask, dummy_mapx[proc_id].reshape(1, size_template_y),
                  dummy_mapy[proc_id].reshape(1, size_template_y), 0, 0, cv::Scalar(0));
        cv::erode(*current_mask, *current_mask, Erosion_mask); // para reduzir ruidos
	}
}

void ba::WarpTPSColor(cv::Mat *ICurColor, cv::Mat *current_warp_color)
{
    int proc_id = omp_get_thread_num();
	cv::remap(*ICurColor, *current_warp_color, dummy_mapx[proc_id].reshape(1, size_template_y),
              dummy_mapy[proc_id].reshape(1, size_template_y), 1, 0, cvScalar(0));
}


// Computing gradients
void ba::WarpGrad(cv::Mat input, cv::Mat& gradx, cv::Mat& grady)
{
	cv::Sobel(input, gradx, CV_32F, 1, 0, 1);
	cv::Sobel(input, grady, CV_32F, 0, 1, 1);
}


// TPS stuff
void ba::DefineCtrlPts()
{	
    std::cout << "Defining control points..." << std::endl;
	// Initializing control point vector
	ctrl_pts_x = (int*) malloc(n_ctrl_pts_x*n_ctrl_pts_y*sizeof(int));
	ctrl_pts_y = (int*) malloc(n_ctrl_pts_x*n_ctrl_pts_y*sizeof(int));

	int offx = cvCeil((double)size_template_x/2);
	int offy = cvCeil((double)size_template_y/2);

	for(int j=0; j<n_ctrl_pts_x; j++)
		for(int i=0; i<n_ctrl_pts_y; i++)
		{
			ctrl_pts_x[i + j*n_ctrl_pts_y] = cvRound((j+1)*size_template_x/(n_ctrl_pts_x+1) - offx);
			ctrl_pts_y[i + j*n_ctrl_pts_y] = cvRound((i+1)*size_template_y/(n_ctrl_pts_y+1) - offy);
		}
}

void ba::TPSPrecomputations()
{
	Ks.setTo(0);

	// TPS Precomputations start here - I will precompute MKinv, which is the matrix 
	// I will multiply with ctrl_points_w to get the warped pixel positions

	// Mounting Matrix 'K'	
	cv::Mat K(n_ctrl_pts+3, n_ctrl_pts+3, CV_32FC1);

	for(int j=0;j<n_ctrl_pts;j++)
	{
		K.at<float>(j, n_ctrl_pts) = 1;
		K.at<float>(j, n_ctrl_pts+1) = (float) ctrl_pts_x[j];
		K.at<float>(j, n_ctrl_pts+2) = (float) ctrl_pts_y[j];

		K.at<float>(n_ctrl_pts,   j) = 1;
		K.at<float>(n_ctrl_pts+1, j) = (float) ctrl_pts_x[j];
		K.at<float>(n_ctrl_pts+2, j) = (float) ctrl_pts_y[j];		
	}

	for(int i=0;i<n_ctrl_pts;i++)
		for(int j=0;j<n_ctrl_pts;j++)
			K.at<float>(i, j) = Tps(Norm( (float) (ctrl_pts_x[i]-ctrl_pts_x[j]), (float) (ctrl_pts_y[i]-ctrl_pts_y[j])));

	for(int i=n_ctrl_pts; i<n_ctrl_pts+3; i++)
		for(int j=n_ctrl_pts; j<n_ctrl_pts+3; j++)
			K.at<float>(i, j) = 0;

	// Inverting Matrix 'K'
	cv::Mat K2 = K.inv(CV_LU);

	// Passing result to Kinv
	for(int i=0;i<n_ctrl_pts+3;i++)
	{
		for(int j=0;j<n_ctrl_pts;j++)
		{
			Kinv.at<float>(i, j) = K2.at<float>(i, j);

			if(i<n_ctrl_pts)
			{
				Ks.at<float>(i, j) = Kinv.at<float>(i, j);
				Ks.at<float>(i+n_ctrl_pts, j+n_ctrl_pts) = Ks.at<float>(i, j);
			}
		}
	}

	// Creating Matrix 'M'	
	int offx = cvCeil((double)size_template_x/2);
	int offy = cvCeil((double)size_template_y/2);

	for(int i=0;i<size_template_y;i++)
	{
		for(int j=0;j<size_template_x;j++)
		{
			for(int k=0;k<n_ctrl_pts;k++)
				M.at<float>(j+size_template_x*i, k) = Tps(Norm((float)(j-offx - ctrl_pts_x[k]), (float)(i-offy - ctrl_pts_y[k])));

			M.at<float>(j+size_template_x*i, n_ctrl_pts) = 1;
			M.at<float>(j+size_template_x*i, n_ctrl_pts+1) = (float) j-offx;
			M.at<float>(j+size_template_x*i, n_ctrl_pts+2) = (float) i-offy;	
		}
	}

	// Pre-computing M with Kinv
	MKinv = M*Kinv;
    cv::Mat temp = MKinv.t();
    (temp).copyTo(MKinvT);
}

float ba::Tps(float r)
{
	float ans;

	if(r != 0)
		ans = r*r*log10(r*r);
	else
		ans = 0;

	return ans;
}

float ba::Norm(float x, float y)
{
	float ans;

	ans = pow(x*x+y*y, 0.5f);

    return ans;
}

void ba::add_noise(cv::Mat ICur[], cv::Mat MaskICur[], cv::Mat ICurColor[])
{
    for(int index=0; index<n_elements; index++) 
        if(!is_anchor(index))
        {
            for(int i = 0; i < n_ctrl_pts; i++)
            {
                ctrl_pts_x_w[index].at<float>(i, 0) += Utils::random(-5, 5);
                ctrl_pts_y_w[index].at<float>(i, 0) += Utils::random(-5, 5);
            }
            WarpTPS(ICurComp[index], &MaskICur[index], current_warp[index],
                    &current_mask[index], ctrl_pts_x_w[index], ctrl_pts_y_w[index]);
            WarpTPSColor(&ICurColor[index], &current_warp_color[index]);
        }

    // TPS Precomputations
    TPSPrecomputations();
}

bool ba::is_anchor(int element_index)
{
    for(std::vector<int>::iterator it = anchors.begin(); it != anchors.end(); it++)
        if(*it == element_index)
            return true;
    return false;
}

int ba::anchors_before(int element_index)
{
    int result = 0;
    for(std::vector<int>::iterator it = anchors.begin(); it != anchors.end(); it++)
        if(*it < element_index)
            result++;
        else
            break;
    return result;
}
