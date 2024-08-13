#include "ba.h"

ba::~ba()
{
    ncc_stats.close();
    time_stats.close();
    delete locks;
    free(ctrl_pts_x);
    free(ctrl_pts_y);
    free(ctrl_pts_xi);
    free(ctrl_pts_yi);
//    ssd_stats.close();
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
                          std::vector<cv::Point> anchors,
                          std::string workspace_name)
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
    this->epsilon = epsilon;
    this->n_max_iters = n_max_iters;
    this->interp = interp;
    this->Visibility_map = Visibility_map;
    this->anchor_positions = anchors;
    this->using_masks = true;
    this->using_colors = true;
    this->size_template = size_template_x*size_template_y;
    this->inter_regularization_weight = 1;
    this->intra_regularization_weight = 3;
    this->min_overlap = 0.3;
//    this->total_time = 0;

    // Open output for stats
    std::stringstream ss;
    ss << "../storage/results_2015_07_17/1_od1_ncc.txt";
    ncc_stats.open(ss.str());
    ss.str("");
    ss << "../storage/results_2015_07_17/1_od1_time.txt";
    time_stats.open(ss.str());
//    ssd_stats.open("../storage/ssd.txt");

    // Process visibility_map
    ProcessVisibilityMap();

    // Process illumination parameters
    InitIllum();

    // Reference pixel list when n_active_pixels = size_template_x*size_template_y
//    std_pixel_list = (int*)malloc(size_template_x*size_template_y*sizeof(int));
//    for(int i=0;i<size_template_x*size_template_y;i++)
//        std_pixel_list[i] = i;
//    active_pixels_r = std_pixel_list;
//    active_pixels_g = std_pixel_list;
//    active_pixels_b = std_pixel_list;

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
        ICurComp[i].create(size_template_y, size_template_x, CV_8U);

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
    current_mask.resize(n_elements);
    current_warp_color.resize(n_elements);

    for(int i=0; i<n_elements; i++)
    {
        // transposto com relacao a orientacao que eu assumi no papel
        SDx[i].create(2*n_ctrl_pts, n_active_pixels, CV_32F);

        gradx[i].create(size_template_y, size_template_x, CV_32F);
        grady[i].create(size_template_y, size_template_x, CV_32F);

        current_warp[i].create(size_template_y, size_template_x, CV_8U);
        current_mask[i].create(size_template_y, size_template_x, CV_8U);
        current_warp_color[i].create(size_template_y, size_template_x, CV_8UC3);
    }

    dif.resize(n_combinations);
    SDw.resize(n_combinations);
    combination_masks.resize(n_combinations);

    for(int i=0; i<n_combinations; i++)
    {
        dif[i].create(n_active_pixels, 1, CV_32F);
        SDw[i].create(2*n_ctrl_pts, n_active_pixels, CV_32F);
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
    Erosion_mask = getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3), cv::Point(1,1));

    locks = new omp_lock_t [n_elements];
    for(int i = 0; i < n_elements; i++)
        omp_init_lock(&locks[i]);
}

// Mosaicking
void ba::Process(std::vector<cv::Mat> ICur, std::vector<cv::Mat> MaskICur,
                 std::vector<cv::Mat> ICurColor)
{
    // Compensating illumination on green and red channels in all templates
//    for(int i = 0; i < n_elements; i++)
//        IlluminationCompensation(ICur[i], MaskICur[i], ICurComp[i]);

    // Aqui eu calculo os T' pela primeira vez para o elemento ancora, ja que ele eh igual a
    // T (T'=T)
    // Note que T' eh chamado de 'current_warp'
    // As componentes x e y ds pontos de controle 'p' sao chamados de ctrl_pts_x_w e ctrl_pts_y_w,
    // respectivamente
    // Ponto 5 - eu uso os vetores acima para calcular as componentes x e y dos pixels 'warpados',
    // para cada template T'
    for(int i = 0; i < anchors.size(); i++)
    {
        current_warp[anchors[i]] = ICur[anchors[i]];
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
                WarpTPS(ICur[index], &MaskICur[index], current_warp[index],
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

        // Crioar aqui a funcao ComputeImageWeights()
        // voce vai atualizar w[block]
        ComputeCombinationMasks();

        // Saving stats to file
        for(int block = 0; block < n_combinations; block++)
        {
            ncc_stats << Utils::ncc_float(current_warp[combinations[block][0]],
                                          current_warp[combinations[block][1]],
                                          combination_masks[block],
                                          combinations[block][3],
                                          combinations[block][2]) << " ";
//            ssd_stats << Utils::ssd_similarity(current_warp_color[combinations[block][0]],
//                                               current_warp_color[combinations[block][1]],
//                                               combination_masks[block],
//                                               combinations[block][2],
//                                               combinations[block][3]) << " ";
        }
        ncc_stats << "\n";
//        ssd_stats << "\n";

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

        double time_wjd = (cvGetTickCount()-cs)/(1000*cvGetTickFrequency());
        std::cout << "Warps, jacobians and differences: " << time_wjd << " ms" << std::endl;
        time_stats << time_wjd << " ";
        cs = cvGetTickCount();

        // nessa funcao, usando as jacobianas de cada template T' previamente calculadas no
        // ponto (7), eu devo mapear a derivada (jacobiana) de cada template com offset da minha
        // lista de correspondencias
        // e colocar o resultado em uma matriz chamada 'SDw'
        // note que eu nao preciso calcular a jacobiana com offset para o template ancora
        // no exemplo do papel, eu nao tive que calcular a jacobiana com offset para o primeiro
        // termo de alpha, ja que T1 era o template ancora
        BuildOffsetJacobians();

        double time_offsets = (cvGetTickCount()-cs)/(1000*cvGetTickFrequency());
        std::cout << "Offsets: " << time_offsets << " ms" << std::endl;
        time_stats << time_offsets << " ";
        cs = cvGetTickCount();

        // O gradiente eh a multiplicacao das jacobianas com as diferencas para cada combinacao na
        // minha lista de combinacoes

        // ponto (10)
        ComputeGradient();

        double time_gradient = (cvGetTickCount()-cs)/(1000*cvGetTickFrequency());
        std::cout << "Gradient: " << time_gradient << " ms" << std::endl;
        time_stats << time_gradient << " ";
        cs = cvGetTickCount();

        // Fazer hessianas
        // esse calculo ainda nao esta bom, precisamos de atualizar essa funcao aqui e seu mestrado
        // estara pronto
        ComputeBigHessian();

        double time_hessian = (cvGetTickCount()-cs)/(1000*cvGetTickFrequency());
        std::cout << "Hessian: " << time_hessian << " ms" << std::endl;
        time_stats << time_hessian << "\n";
        cs = cvGetTickCount();

        if(UpdateParams())
        {
            std::cout << "Numero de iteracoes: " << iterations << std::endl;
            break;
        }

        std::cout << "Everything: " << time_wjd + time_offsets + time_gradient + time_hessian
                  << " ms" << std::endl;

//        total_time += (cvGetTickCount()-cs)/(1000*cvGetTickFrequency());

        // Display
        std::cout << ">> End of iteration: " << iterations << std::endl << std::endl;

        // Show all images in a grid (only works for Lena).
//        std::stringstream ss;
//        int index = 0, size = 154;
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

//        for(int index=0; index<n_elements; index++)
//        {
//            char text[30];
//            sprintf(text, "t'%d", index);
//            cv::imshow(text, current_warp_color[index]);
//        }
//        cv::waitKey(10);
    }

    std::stringstream ss;
    for(int i = 0; i < n_elements; i++)
    {
        ss << "../output/" << i << ".png";
        cv::imwrite(ss.str(), current_warp_color[i]);
        ss.str("");
    }
}

// Misc
int    ba::UpdateParams()
{
    // ponto 14
    cv::Mat delta = hessian.inv()*gradient;

    // Update
    int check = 0, offset = 0;
    int n_inactive_templates = inactive_templates();

    for(int index = 0; index < n_elements; index++)
    {
        if(is_inactive(index))
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

    std::cout << check << " converged out of " << n_elements - n_inactive_templates <<  std::endl;

    return check == n_elements - n_inactive_templates;
}

cv::Mat ba::ComputeHessian(cv::Mat* Jacobian1, cv::Mat *Jacobian2, cv::Mat mask)
{
    cv::Mat result = cv::Mat::zeros(2*n_ctrl_pts, 2*n_ctrl_pts, CV_32F);

    // Computing dot products
    #pragma omp parallel for
    for(int i=0;i<2*n_ctrl_pts;i++)
        for(int j=i;j<2*n_ctrl_pts;j++)
            for(int k=0;k<Jacobian1->cols;k++)
                if(mask.data[k] != 0)
                    result.at<float>(i,j) += Jacobian1->at<float>(i,k) *
                                             Jacobian2->at<float>(j,k);

    // Reflecting Hessians
    #pragma omp parallel for
    for(int i=0;i<2*n_ctrl_pts;i++)
        for(int j=i;j<2*n_ctrl_pts;j++)
            result.at<float>(j,i) = result.at<float>(i,j);

    return result;
}


void    ba::ComputeBigHessian()
{
    // Clean hessian
    hessian = cv::Mat::zeros(2*n_ctrl_pts*(n_active_templates - n_anchors),
                             2*n_ctrl_pts*(n_active_templates - n_anchors), CV_32F);
    int offset = 2*n_ctrl_pts;

    // Filling intra_regulator with constants resulting from second derivation
    float intra_constant = intra_regularization_weight * 2;
    cv::Mat intra_regulator = cv::Mat::zeros(2*n_ctrl_pts, 2*n_ctrl_pts, CV_32F);
    for(int i = 0; i < intra_regulator.rows; i++)
        intra_regulator.at<float>(i, i) = 2;
    int minus_two_rows[16] = {0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7};
    int minus_two_cols[16] = {1, 2, 0, 3, 0, 3, 1, 2, 5, 6, 4, 7, 4, 7, 5, 6};
    for(int i = 0; i < 16; i++)
        intra_regulator.at<float>(minus_two_rows[i], minus_two_cols[i]) = -1;

    for(int i = 0; i < n_elements; i++)
        if(!is_inactive(i))
        {
            int index = i - inactive_templates_before(i);
            hessian(cv::Rect(index*offset, index*offset, offset, offset)) += intra_constant *
                                                                             intra_regulator;
        }

    // nesse primeiro momento, eu calculo as componentes parcial alpha / (parcial p0, p0) , parcial alpha / (parcial p1, p1) ...
    // que sao as diagonais da matriz H indicada no ponto (23)
    // O problema aqui ´e que eu nao levo em consideracao os pesos w de cada combinacao

    // for every pair of correspondences
    for(int block = 0; block < n_combinations; block++)
    {
        int first = combinations[block][0];
        int second = combinations[block][1];
        if(combination_states[block] == ACTIVE)
        {
            // Fixing index because anchor template is not in big hessian
            int hessian_index_1 = first - inactive_templates_before(first);
            int hessian_index_2 = second - inactive_templates_before(second);

            float regularization_value = inter_regularization_weight * ncc_weight * 2;

            // aqui eu devo calcular a hessiana correspondente ao primeiro elemento da combinacao 'block'
            // como faço isso? H = J^t J, onde cada J leva em consideracao os pesos w[block] (veja folha 16)
            if(!is_anchor(first))
            {
                cv::Mat little_hessian = ComputeHessian(&SDx[first], &SDx[first], combination_masks[block]);
                for(int i = 0; i < n_ctrl_pts; i++) // Regularization term
                    little_hessian.at<float>(i, i) += regularization_value;
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
                for(int i = 0; i < n_ctrl_pts; i++) // Regularization term
                    little_hessian.at<float>(i, i) += regularization_value;
                hessian(cv::Rect(hessian_index_2*offset, hessian_index_2*offset, offset, offset)) += little_hessian; // ponto (24)
            }

            // Aqui eu faço a hessiana usando as jacobianas do primeiro template (SDx) e do segundo
            // template warpado (SDw)
            // note que eu tambem nao levo em conta os pesos w[block]
            if(!is_anchor(first) && !is_anchor(second))
            {
                cv::Mat little_hessian = ComputeHessian(&SDx[first], &SDw[block],
                                                        combination_masks[block]);
                for(int i = 0; i < n_ctrl_pts; i++) // Regularization term
                    little_hessian.at<float>(i, i) += regularization_value;
                hessian(cv::Rect(hessian_index_1*offset, hessian_index_2*offset, offset, offset)) -= little_hessian; // ponto (24)
            }
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
    #pragma omp parallel for
    for(int block = 0; block < n_combinations; block++)
    {
        int second = combinations[block][1];
        if(!is_anchor(second) && combination_states[block] == ACTIVE)
        {
            // I take the offset from the second template
            int offset_x = combinations[block][2];
            int offset_y = combinations[block][3];

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
                        SDw[block].at<float>(k, index) = SDx[second].at<float>(k, index2);
                    else
                        SDw[block].at<float>(k, index) = 0;
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
    n_active_templates = n_elements;
    connections.resize(n_elements);
    for(int i = 0; i < n_elements; i++)
        connections[i] = 0;

    // Alocação do vetor de combinações
    combinations.resize(10000);
    for(int i = 0; i < 10000; i++)
    {
        combinations[i].resize(4);
        combinations[i][0] = 0;
    }

    n_combinations = 0;

    // Max offsets that guarantee a 50% overlap
//    int max_offset_x = size_template_x / (2 * offset_templates);
//    int max_offset_y = size_template_y / (2 * offset_templates);
//    int max_offset_diagonal = size_template_x + size_template_y +
//        sqrt(size_template_x*size_template_x + size_template_y*size_template_y) / 2*offset_templates;

    //TODO fix
    int max_offset_x = 6;
    int max_offset_y = 6;
    int max_offset_diagonal = 4;

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
                        int first = Element_number.at<int>(i, j);
                        int second = Element_number.at<int>(i + offset, j);
                        std::vector<int> combination;
                        combination.push_back(first);
                        combination.push_back(second);
                        combination.push_back(0);
                        combination.push_back(offset * offset_templates);
                        combinations[n_combinations] = combination;
                        connections[first]++;
                        connections[second]++;
                        n_combinations++;
                        break;
                    }
                }

                // Closest neighbor to the right
                for(int offset = 1; j + offset < grid_x && offset <= max_offset_x; offset++)
                {
                    if(Visibility_map.at<uchar>(i, j + offset) != 0)
                    {
                        int first = Element_number.at<int>(i, j);
                        int second = Element_number.at<int>(i, j + offset);
                        std::vector<int> combination;
                        combination.push_back(first);
                        combination.push_back(second);
                        combination.push_back(offset * offset_templates);
                        combination.push_back(0);
                        combinations[n_combinations] = combination;
                        connections[first]++;
                        connections[second]++;
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
                        int first = Element_number.at<int>(i, j);
                        int second = Element_number.at<int>(i + offset, j - offset);
                        std::vector<int> combination;
                        combination.push_back(first);
                        combination.push_back(second);
                        combination.push_back(-offset * offset_templates);
                        combination.push_back(offset * offset_templates);
                        combinations[n_combinations] = combination;
                        connections[first]++;
                        connections[second]++;
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
                        int first = Element_number.at<int>(i, j);
                        int second = Element_number.at<int>(i + offset, j + offset);
                        std::vector<int> combination;
                        combination.push_back(first);
                        combination.push_back(second);
                        combination.push_back(offset * offset_templates);
                        combination.push_back(offset * offset_templates);
                        combinations[n_combinations] = combination;
                        connections[first]++;
                        connections[second]++;
                        n_combinations++;
                        break;
                    }
                }
            }
        }

    combination_states.resize(n_combinations);
    for(int i = 0; i < n_combinations; i++)
        combination_states[i] = ACTIVE;

    std::cout << " Here are all possible combinations:" << std::endl;
    for(int i=0; i<n_combinations; i++)
    {
//        std::cout << combinations[i][0] << " " << combinations[i][1] << " " << combinations[i][2]
//                  << " " << combinations[i][3] << std::endl;
        ncc_stats << combinations[i][0] << " ";
//        ssd_stats << combinations[i][0] << " ";
    }
    ncc_stats << "\n";
//    ssd_stats << "\n";

    for(int i=0; i<n_combinations; i++)
    {
        ncc_stats << combinations[i][1] << " ";
//        ssd_stats << combinations[i][1] << " ";
    }
    ncc_stats << "\n";
//    ssd_stats << "\n";
}


// *** nesta funcao eu estou adicionando cada diferenca multiplicada pelas jacobianas apropriadas
// (SDx e SDw, para o primeiro e segundo elementos da combinacao)
// nas colunas correspondntes de G (ponto 12, atras da pagina 6)
void    ba::ComputeGradient()
{
    // esse vetor corresponde ao vetor G do ponto (11) e ponto (10)
    // Resetting gradient vector
    gradient = cv::Mat::zeros(2*n_ctrl_pts*(n_active_templates-n_anchors), 1, CV_32F);

    float intra_constant = intra_regularization_weight * 2;
    for(int i = 0; i < n_elements; i++)
    {
        if(!is_inactive(i))
        {
            // Control points
            float p0x = ctrl_pts_x_w[i].at<float>(0, 0);
            float p1x = ctrl_pts_x_w[i].at<float>(1, 0);
            float p2x = ctrl_pts_x_w[i].at<float>(2, 0);
            float p3x = ctrl_pts_x_w[i].at<float>(3, 0);
            float p0y = ctrl_pts_y_w[i].at<float>(0, 0);
            float p1y = ctrl_pts_y_w[i].at<float>(1, 0);
            float p2y = ctrl_pts_y_w[i].at<float>(2, 0);
            float p3y = ctrl_pts_y_w[i].at<float>(3, 0);

            cv::Mat intra_regulator(2*n_ctrl_pts, 1, CV_32F);
            intra_regulator.at<float>(0, 0) = 2*p0x - p1x - p2x + ctrl_pts_offset_x;
            intra_regulator.at<float>(1, 0) = -p0x + 2*p1x - p3x + ctrl_pts_offset_x;
            intra_regulator.at<float>(2, 0) = -p0x + 2*p2x - p3x - ctrl_pts_offset_x;
            intra_regulator.at<float>(3, 0) = -p1x - p2x + 2*p3x - ctrl_pts_offset_x;
            intra_regulator.at<float>(4, 0) = 2*p0y - p1y -p2y + ctrl_pts_offset_y;
            intra_regulator.at<float>(5, 0) = -p0y + 2*p1y - p3y - ctrl_pts_offset_y;
            intra_regulator.at<float>(6, 0) = -p0y + 2*p2y - p3y + ctrl_pts_offset_y;
            intra_regulator.at<float>(7, 0) = -p1y - p2y + 2*p3y - ctrl_pts_offset_y;
            int gradient_index = i - inactive_templates_before(i);
            omp_set_lock(&locks[i]);
            gradient(cv::Rect(0,gradient_index*2*n_ctrl_pts, 1, 2*n_ctrl_pts)) += intra_constant *
                                                                                  intra_regulator;
            omp_unset_lock(&locks[i]);
        }
    }

    // no loop abaixo, eu vou percorrer cada combinacao.
    // para o primeiro e segundo elemento da combinacao, eu vou calcular o vetor de gradiente
    // 'combination_gradient' para cada
    // elemento separadamente e adiciona-lo na posicao apropriada em G (vetor 'gradient')
    #pragma omp parallel for
    for(int block = 0; block < n_combinations; block++)
    {
        if(combination_states[block] == ACTIVE)
        {
            int first = combinations[block][0];
            int second = combinations[block][1];

            int index_column1 = first - inactive_templates_before(first);
            int index_column2 = second - inactive_templates_before(second);

            ncc_weight = 1 - Utils::ncc_float(current_warp[first], current_warp[second],
                                              combination_masks[block], combinations[block][3],
                                              combinations[block][2]);
            float inter_constant = inter_regularization_weight * ncc_weight * 2;
            cv::Mat inter_regulator(2*n_ctrl_pts, 1, CV_32F);

            // For every correspondence, we compute the product between Jacobians and image difference
            if(!is_anchor(first))
            {
                // note que SDx foi montado transposto de proposito, para essa operacao aqui
                // note tambem coitado, que voce pode melhorar essa operacao divindo ainda mais
                // os calculos entre os processadores da sua maquina, usando openmp
                cv::Mat combination_gradient = SDx[first]*dif[block];

                for(int i = 0; i < n_ctrl_pts; i++)
                {
                    inter_regulator.at<float>(i, 0) = inter_constant *
                                                      (ctrl_pts_x_w[first].at<float>(i, 0) -
                                                      ctrl_pts_x_w[second].at<float>(i, 0));
                    inter_regulator.at<float>(i+n_ctrl_pts, 0) = inter_constant *
                                                           (ctrl_pts_y_w[first].at<float>(i, 0) -
                                                           ctrl_pts_y_w[second].at<float>(i, 0));
                }
                combination_gradient += inter_regulator;

                // Em seguida eu incremento a posicao no vetor 'gradient'
                omp_set_lock(&locks[first]);
                gradient(cv::Rect(0,index_column1*2*n_ctrl_pts, 1, 2*n_ctrl_pts)) +=
                    combination_gradient;
                omp_unset_lock(&locks[first]);
            }

            if(!is_anchor(second))
            {
                // note que aqui eu uso a jacobiana warpada do template T'
                // e note tambem que a jacobiana warpada eh diferente para cada combinacao, ja que o
                // offset muda para cada comb.
                cv::Mat combination_gradient = -SDw[block]*dif[block];  // ponto (13)

                for(int i = 0; i < n_ctrl_pts; i++)
                {
                    inter_regulator.at<float>(i, 0) = -inter_constant *
                                                      (ctrl_pts_x_w[first].at<float>(i, 0) -
                                                      ctrl_pts_x_w[second].at<float>(i, 0));
                    inter_regulator.at<float>(i+n_ctrl_pts, 0) = -inter_constant *
                                                           (ctrl_pts_y_w[first].at<float>(i, 0) -
                                                           ctrl_pts_y_w[second].at<float>(i, 0));
                }
                combination_gradient += inter_regulator;

                // Em seguida eu incremento a posicao no vetor 'gradient'
                omp_set_lock(&locks[second]);
                gradient(cv::Rect(0,index_column2*2*n_ctrl_pts,1,2*n_ctrl_pts)) +=
                    combination_gradient;
                omp_unset_lock(&locks[second]);
            }
        }
    }
}

void ba::ComputeCombinationMasks()
{
//    #pragma omp parallel for
    for(int pair = 0; pair < n_combinations; pair++)
    {
        if(combination_states[pair] == ACTIVE)
        {
            int first = combinations[pair][0];
            int second = combinations[pair][1];
            int col_offset = combinations[pair][2];
            int row_offset = combinations[pair][3];
            cv::Mat mask_0 = current_mask[first];
            cv::Mat mask_1 = current_mask[second];

            for(int row = 0; row < size_template_y; row++)
            {
                for(int col = 0; col < size_template_x; col++)
                {
                    if(row >= row_offset &&
                       col >= col_offset &&
                       (col - col_offset) < size_template_x &&
                       (row - row_offset) < size_template_y)
                    {
                        int index = row * size_template_x + col;
                        combination_masks[pair].at<uchar>(index, 0) = mask_0.at<uchar>(row, col) &&
                            mask_1.at<uchar>(row-row_offset, col-col_offset);
                    }
                }
            }
            if(Utils::active_area(combination_masks[pair]) <
               combination_masks[pair].total() * min_overlap)
            {
                combination_states[pair] = INACTIVE;
                connections[first]--;
                if(connections[first] == 0)
                    n_active_templates--;
                connections[second]--;
                if(connections[second] == 0)
                    n_active_templates--;
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
            // Active pixel positions
            int i = cvFloor((float)index/size_template_x);
            int j = index - i*size_template_x;

            // If it's an active pixel
            // ponto (21)
            // isso aqui esta realmente errado/incompleto
//            if(current_mask[element].at<uchar>(i,j) != 0) // nao deveria haver check da mascara neste ponto, pois ela j´a vai ser levada em
                                                            // consideracao quando eu for gerar o gradiente
            {
                // Build Jacobian image 1
                // 'x' derivative
                SDx[element].at<float>(k, index) = MKinvT.at<float>(k,index)*gradx[element].at<float>(i,j);

                // 'y' derivative
                SDx[element].at<float>(k+n_ctrl_pts, index) = MKinvT.at<float>(k,index)*grady[element].at<float>(i,j);
            }
//            else
//            {
//                // Pointer to J and M is updated at every increment of index
//                SDx[element].at<float>(k, index) = 0;
//                SDx[element].at<float>(k+n_ctrl_pts, index) = 0;
//            }
        }
}

void ba::ComputeImageDifference()
{
    //  como eu faco isso de maneira automatizada, eu processo cada entrada da minha lista de correspondencias,
    // descubro qual sao os T' pareados e leio o offset entre eles. Os indices do par de templates eh determinado
    // em pos_0 e pos_1, ok?
    // Em seguida, o offset de pos_1 eh determinado como offset_x e y

    // For every pair
    #pragma omp parallel for
    for(int block = 0; block < n_combinations; block++)
    {
        if(combination_states[block] == ACTIVE)
        {
            int first = combinations[block][0];
            int second = combinations[block][1];
            int offset_x = combinations[block][2];
            int offset_y = combinations[block][3];

            // Computing pixel difference
            for(int index=0; index<n_active_pixels; index++)  // lembre-se que n_active_pixels quer dizer simplesmente numero de pixels na imagem T'
            {
                int i = cvFloor((float)index/size_template_x);
                int j = index - i*size_template_x;

                // nestas 4 linhas, eu defino os boundaries de A, o conjunto de pixels de interseccao
                // entre T'[first] e T'[second]
    //            if(i>=offset_y &&
    //               j>=offset_x && // Ponto (8)
    //               (j-offset_x) < size_template_x &&
    //               (i-offset_y) < size_template_y)

                // ponto (22) - TODO - atualmente, eu so desativo os pixels qu enao estao na
                // interseccao. Aqui vou ter que desativar os pixels que estao desativados em
                // current_mask[first] e
                // current_mask[second] (com offset) lembre-se rodrigo, com offset apropriado, por
                // favor , nao crie mais bugs que o necessario, claro que faz a diferenca
                if(combination_masks[block].data[index] != 0)
    //                dif[block].at<float>(index, 0) = (float)current_warp[first].at<uchar>(i,j) -
    //                    (float)current_warp[second].at<uchar>(i-offset_y,j-offset_x); // ponto (7)
                    // For gradient images
                    dif[block].at<float>(index, 0) = current_warp[first].at<float>(i,j) -
                        current_warp[second].at<float>(i-offset_y,j-offset_x); // ponto (7)
                else
                    dif[block].at<float>(index, 0) = 0;
            }
        }
    }
}

void ba::WarpTPS(cv::Mat ICur, cv::Mat* Mask_roi, cv::Mat& current_warp, cv::Mat* current_mask,
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
    ctrl_pts_offset_x = size_template_x/(n_ctrl_pts_x+1);
    ctrl_pts_offset_y = size_template_y/(n_ctrl_pts_y+1);

	for(int j=0; j<n_ctrl_pts_x; j++)
		for(int i=0; i<n_ctrl_pts_y; i++)
		{
			ctrl_pts_x[i + j*n_ctrl_pts_y] = cvRound((j+1)*ctrl_pts_offset_x - offx);
			ctrl_pts_y[i + j*n_ctrl_pts_y] = cvRound((i+1)*ctrl_pts_offset_y - offy);
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

//void ba::add_noise(cv::Mat ICur[], cv::Mat MaskICur[], cv::Mat ICurColor[])
//{
//    for(int index=0; index<n_elements; index++) 
//        if(!is_anchor(index))
//        {
//            for(int i = 0; i < n_ctrl_pts; i++)
//            {
//                ctrl_pts_x_w[index].at<float>(i, 0) += Utils::random(-5, 5);
//                ctrl_pts_y_w[index].at<float>(i, 0) += Utils::random(-5, 5);
//            }
//            WarpTPS(ICurComp[index], &MaskICur[index], current_warp[index],
//                    &current_mask[index], ctrl_pts_x_w[index], ctrl_pts_y_w[index]);
//            WarpTPSColor(&ICurColor[index], &current_warp_color[index]);
//        }
//
//    // TPS Precomputations
//    TPSPrecomputations();
//}

bool ba::is_anchor(int element_index)
{
    for(std::vector<int>::iterator it = anchors.begin(); it != anchors.end(); it++)
        if(*it == element_index)
            return true;
    return false;
}

bool ba::is_inactive(int index)
{
    return is_anchor(index) || connections[index] == 0;
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

int ba::inactive_templates_before(int index)
{
    if(index == 0)
        return 0;

    int result = 0;
    for(int i = 0; i < index; i++)
        if(connections[i] == 0)
            result++;
        else if(is_anchor(i))
            result++;
    return result;
}

int ba::inactive_templates()
{
    int result = 0;
    for(int i = 0; i < n_elements; i++)
        if(connections[i] == 0)
            result++;
        else if(is_anchor(i))
            result++;
    return result;
}
