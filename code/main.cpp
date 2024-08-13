#include <stdio.h>
#include "ba.h"
#include "tracking_aux.h"
#include "MOSAIC.h"

int start_x, start_y;
void selection_callback(int event, int x, int y, int flags, void* data)
{
    cv::Mat* source = (cv::Mat*) data;
    if(event == cv::EVENT_LBUTTONDOWN)
    {
        start_x = x;
        start_y = y;
//        cv::Vec3b color = source->at<cv::Vec3b>(y / 10, x / 10);
//        if(color == cv::Vec3b(255, 255, 255))
//            source->at<cv::Vec3b>(y / 10, x / 10) = cv::Vec3b(0, 0, 255);
//        else if(color == cv::Vec3b(0, 0, 255))
//            source->at<cv::Vec3b>(y / 10, x / 10) = cv::Vec3b(255, 255, 255);
    }
    if(event == cv::EVENT_LBUTTONUP)
    {
        for(int row = start_y / 10; row < y / 10 + 1; row++)
            for(int col = start_x / 10; col < x / 10 + 1; col++)
                if(source->at<cv::Vec3b>(row, col) == cv::Vec3b(255, 255, 255))
                    source->at<cv::Vec3b>(row, col) = cv::Vec3b(0, 0, 255);
    }
}

cv::Mat selection_interface(cv::Mat source, std::vector<cv::Point> pre_selected)
{
    cv::Vec3b red = cv::Vec3b(0, 0, 255);
    cv::Vec3b blue = cv::Vec3b(255, 0, 0);
    cv::Vec3b white = cv::Vec3b(255, 255, 255);
    cv::Mat color_source, big_source;
    cvtColor(source * 255, color_source, CV_GRAY2BGR);
    for(int i = 0; i < pre_selected.size(); i++)
    {
        cv::Point p = pre_selected[i];
        color_source.at<cv::Vec3b>(p.x, p.y) = blue;
    }
    resize(color_source, big_source, source.size() * 10, 0, 0, cv::INTER_NEAREST);
    cv::namedWindow("Selection");
    cv::setMouseCallback("Selection", selection_callback, &color_source);
    char key = 0;
    while(key != 'q')
    {
        resize(color_source, big_source, source.size() * 10, 0, 0, cv::INTER_NEAREST);
        cv::imshow("Selection", big_source);
        key = cv::waitKey(150);
    }
    cv::destroyWindow("Selection");

    cv::Mat result = cv::Mat::zeros(color_source.size(), CV_8U);
    for(int row = 0; row < color_source.rows; row++)
        for(int col = 0; col < color_source.cols; col++)
            if(color_source.at<cv::Vec3b>(row, col) == red ||
               color_source.at<cv::Vec3b>(row, col) == blue)
//               || color_source.at<cv::Vec3b>(row, col) == white)
                result.at<uchar>(row, col) = 1;
    return result;
}

cv::Mat apply_mask(cv::Mat image, cv::Mat mask) {
    cv::Mat result = cv::Mat::zeros(image.size(), image.type());
    for(int r = 0; r < image.rows; r++)
        for(int c = 0; c < image.cols; c++)
            if(mask.at<uchar>(r, c) != 0)
                result.at<cv::Vec3b>(r, c) = image.at<cv::Vec3b>(r, c);
    return result;
}

int main(void)
{
    omp_set_num_threads(NUM_THREADS);

	// ba parameters
	std::string	filename;

    int		size_template_x,
			size_template_y,
			size_bins,
			percentage_active,
			n_bins,
			n_max_iters,
            interp,
            n_ctrl_pts_x,
            n_ctrl_pts_y,
            n_ctrl_pts_xi,
            n_ctrl_pts_yi;

	int		grid_x,
			grid_y,
			offset_templates;

	float	lambda,
			epsilon,
			confidence_thres,
            confidence;

	double	clock_start,
			clock_stop,
			tick = cvGetTickFrequency();
			
//    cv::Mat *ICur,
//            *ICurColor,
//            *MaskICur,
//            Visibility_map;
    cv::Mat Visibility_map;
    std::vector<cv::Mat> ICur, ICurColor, MaskICur;
	
	ba	 mosaicker; // main bundle adjustment class

    MOSAIC	refine_mosaic1,
            refine_mosaic2;

    std::vector<cv::Point> anchors;

    // Loading ba parameters from file
    LoadParameters("../settings/parameters.yml",
                   &grid_x,
                   &grid_y,
                   &offset_templates,
                   &size_template_x,
                   &size_template_y,
                   &n_ctrl_pts_x,
                   &n_ctrl_pts_y,
                   &n_ctrl_pts_xi,
                   &n_ctrl_pts_yi,
                   &lambda,
                   &size_bins,
                   &epsilon,
                   &n_bins,
                   &n_max_iters,
                   &confidence_thres,
                   &percentage_active,
                   &filename,
                   &interp);

    // Loads workspace from file
    std::cout << " Loading workspace ... " << std::endl;
    cv::FileStorage fs("../storage/workspaces/1_od1.yml", cv::FileStorage::READ);
//    cv::FileStorage fs("../storage/workspaces/3_od2.yml", cv::FileStorage::READ);
//    cv::FileStorage fs("../storage/workspaces/3_od3.yml", cv::FileStorage::READ);
//    cv::FileStorage fs("../storage/workspaces/3_od4.yml", cv::FileStorage::READ);
//    cv::FileStorage fs("../storage/workspaces/3_od5.yml", cv::FileStorage::READ);
//    cv::FileStorage fs("../storage/workspaces/3_og1.yml", cv::FileStorage::READ);
//    cv::FileStorage fs("../storage/workspaces/3_og2.yml", cv::FileStorage::READ);
//    cv::FileStorage fs("../storage/workspaces/3_og4.yml", cv::FileStorage::READ);
//    cv::FileStorage fs("../storage/workspaces/3_og5.yml", cv::FileStorage::READ);
//    cv::FileStorage fs("../storage/workspaces/3_og7.yml", cv::FileStorage::READ);
//    cv::FileStorage fs("../storage/workspaces/4_od2.yml", cv::FileStorage::READ);
//    cv::FileStorage fs("../storage/workspaces/4_od3.yml", cv::FileStorage::READ); //narrow slit
//    cv::FileStorage fs("../storage/workspaces/4_od5.yml", cv::FileStorage::READ);
//    cv::FileStorage fs("../storage/workspaces/4_od6.yml", cv::FileStorage::READ); //bad gradient
//    cv::FileStorage fs("../storage/workspaces/4_od8.yml", cv::FileStorage::READ);
//    cv::FileStorage fs("../storage/workspaces/4_od10.yml", cv::FileStorage::READ);
//    cv::FileStorage fs("../storage/workspaces/4_og2.yml", cv::FileStorage::READ);
//    cv::FileStorage fs("../storage/workspaces/4_og5.yml", cv::FileStorage::READ);
//    cv::FileStorage fs("../storage/workspaces/4_og6.yml", cv::FileStorage::READ);
//    cv::FileStorage fs("../storage/workspaces/4_og9.yml", cv::FileStorage::READ);
//    cv::FileStorage fs("../storage/workspaces/5_od1.yml", cv::FileStorage::READ);
//    cv::FileStorage fs("../storage/workspaces/5_od2.yml", cv::FileStorage::READ);
//    cv::FileStorage fs("../storage/workspaces/6_od1.yml", cv::FileStorage::READ);
//    cv::FileStorage fs("../storage/workspaces/6_od2.yml", cv::FileStorage::READ);
//    cv::FileStorage fs("../storage/workspaces/6_og1.yml", cv::FileStorage::READ);
//    cv::FileStorage fs("../storage/workspaces/6_og2.yml", cv::FileStorage::READ);
//    cv::FileStorage fs("../storage/workspaces/6_og5.yml", cv::FileStorage::READ);
//    cv::FileStorage fs("../storage/workspaces/7_od1.yml", cv::FileStorage::READ);
//    cv::FileStorage fs("../storage/workspaces/7_od6.yml", cv::FileStorage::READ);
//    cv::FileStorage fs("../storage/workspaces/7_og1.yml", cv::FileStorage::READ);
//    cv::FileStorage fs("../storage/workspaces/7_og3.yml", cv::FileStorage::READ);
//    cv::FileStorage fs("../storage/workspaces/7_og5.yml", cv::FileStorage::READ);
//    cv::FileStorage fs("../storage/workspaces/7_og6.yml", cv::FileStorage::READ);
//    cv::FileStorage fs("../storage/workspaces/8_od1.yml", cv::FileStorage::READ);
//    cv::FileStorage fs("../storage/workspaces/8_od2.yml", cv::FileStorage::READ);
//    cv::FileStorage fs("../storage/workspaces/8_od3.yml", cv::FileStorage::READ);
//    cv::FileStorage fs("../storage/workspaces/8_og3.yml", cv::FileStorage::READ);
//    cv::FileStorage fs("../storage/workspaces/8_og5.yml", cv::FileStorage::READ);
//    cv::FileStorage fs("../storage/workspaces/9_od4_single.yml", cv::FileStorage::READ);
//    cv::FileStorage fs("../storage/workspaces/9_od7.yml", cv::FileStorage::READ);
//    cv::FileStorage fs("../storage/workspaces/9_og8_single.yml", cv::FileStorage::READ);

    if(!fs.isOpened())
    {
        printf("\n Could not load workspace.yml!!!\n");
        exit(1);
    }
    else
        std::cout << " Workspace loaded ! " << std::endl;

    fs["VisibilityMap"] >> Visibility_map;

    anchors.push_back(cv::Point(grid_y / 2, grid_x / 2));
    Visibility_map = selection_interface(Visibility_map, anchors);

//    ICur = new cv::Mat [grid_x*grid_y];
//    ICurColor = new cv::Mat [grid_x*grid_y];
//    MaskICur = new cv::Mat [grid_x*grid_y];
    ICur.resize(grid_x*grid_y);
    ICurColor.resize(grid_x*grid_y);
    MaskICur.resize(grid_x*grid_y);

    int counter = 0;
    for(int v = 0; v < Visibility_map.cols; v++)
    {
        for(int u = 0; u < Visibility_map.rows; u++)
        {
            std::ostringstream text;

            if(Visibility_map.at<uchar>(u,v) != 0)
            {
                // Loading templates
                text.str("");
                text << "Template" << u+grid_y*v;
                fs[text.str()] >> ICurColor[counter];

                cv::cvtColor(ICurColor[counter], ICur[counter], CV_BGR2GRAY);
                ICur[counter].convertTo(ICur[counter], CV_32F);
                ICur[counter] = Utils::gradient_image(ICur[counter]);

//                text.str("");
//                text << "../output/" << counter << "_source.png";
//                cv::imwrite(text.str(), ICurColor[counter]);
//
//                text.str("");
//                text << "../output/" << counter << "_grad.png";
//                cv::imwrite(text.str(), ICur[counter]);

                text.str("");
                text << "Mask" << u+grid_y*v;
                fs[text.str()] >> MaskICur[counter];

                counter++;
            }
        }
    }

//    // Using artificial data - Lena image
//    // Loading workspace
//    // Test workspace
//    grid_x = 3;
//    grid_y = 3;
//    size_template_x = size_template_y = 154;
//
//    Visibility_map.create(grid_y, grid_x, CV_8UC1);
//    Visibility_map.setTo(1);
//    anchors.push_back(cv::Point(grid_y / 2, grid_x / 2));
//
//    ICur = new cv::Mat [grid_x*grid_y];
//    ICurColor = new cv::Mat [grid_x*grid_y];
//    MaskICur = new cv::Mat [grid_x*grid_y];
//
//    // Loading images and creating masks
//    int index = 0;
//    for(int col = 0; col < grid_x; col++)
//    {
//        for(int row = 0; row < grid_y; row++)
//        {
//            if(Visibility_map.at<uchar>(row, col) == 1)
//            {
//                cv::Mat lena = cv::imread("../lena.jpg");
////                if(col == grid_x / 2 && row == grid_y / 2)
//                    ICurColor[index] = lena;
////                else
////                {
////                    // Applying offset
////                    cv::Mat warp = cv::Mat::zeros(2, 3, CV_32F);
////                    warp.at<float>(0, 0) = 1;
////                    warp.at<float>(0, 2) = (grid_x/2 - col) * offset_templates;
////                    warp.at<float>(1, 1) = 1;
////                    warp.at<float>(1, 2) = (grid_y/2 - row) * offset_templates;
////                    warpAffine(lena, ICurColor[index], warp, lena.size(), cv::INTER_NEAREST);
////                }
//
//                cv::cvtColor(ICurColor[index], ICur[index], CV_BGR2GRAY);
//                MaskICur[index].create(ICur[index].cols, ICur[index].rows, CV_8UC1);
//                MaskICur[index].setTo(1);
//                index++;
//            }
//        }
//    }

    // Are we using active pixels?
    int	n_active_pixels = size_template_x*size_template_y; // nao estamos usando pixels ativos

	// Initialize ba structure
//    char key = 0;
//    while(key != 'q')
//    {
//        if(key == 0)
//        {
            std::cout << " Initializing BA structure ... " << std::endl;
            mosaicker.InitializeMosaic(grid_x,
                                       grid_y,
                                       offset_templates,
                                       size_template_x, 
                                       size_template_y,
                                       n_active_pixels,
                                       n_ctrl_pts_x,
                                       n_ctrl_pts_y,
                                       n_ctrl_pts_xi,
                                       n_ctrl_pts_yi,
                                       n_max_iters,
                                       epsilon,
                                       interp,
                                       Visibility_map,
                                       anchors);
//        }
//        else
//            mosaicker.add_noise(ICur, MaskICur, ICurColor);

        // Now initialize mosaic refinement structure
        // Esta ´e a estrutura de criacao da imagem de mosaico B (so pra display)
        refine_mosaic1.Initialize(size_template_x, size_template_y, grid_x, grid_y,
                                  offset_templates, ICurColor, MaskICur, &Visibility_map);

        // Run mosaic refinement
        // novamente s´o display
        refine_mosaic1.Process();

        // Prints mosaic
        // salvo a imagem B em um arquivo
        refine_mosaic1.PrintMosaic("/home/rodrigolinhares/code/bundle_adjustment/code/storage/results_2015_07_17/1_od1.png");

        // Run bundle adjustment
        clock_start = (double)cvGetTickCount();	// Clock start

        std::cout << " Running BA ... " << std::endl;
        // o essencial comeca aqui
        // eu alimento o bundle adjuster com as imagens T, as mascaras, e suas posicoes no mosaico, que
        // foram determinados anteriormente na matriz
        // Visibility_map, no initialize mosaic
        // exemplo: se a minha visibility map fosse
        // [0 0 0 1 0 0]
        // [0 1 1 0 0 0], os templates T0, T1, T2 estariam em
        // [0  0  0 T0 0 0]
        // [0 T1 T2  0 0 0]  <- column major
        mosaicker.Process(ICur, MaskICur, ICurColor);
    
        // Clock start/stop
        clock_stop = ((double)cvGetTickCount()-clock_start)/(1000000*tick);

        // Terminal
        printf("> Elapsed: %f s\n | NCC:", clock_stop);	

        // Now I have to hack something to make the mosaic work
    //    for(int i=0; i<mosaicker.GetNumberElements(); i++)
    //        cv::cvtColor(mosaicker.current_warp[i], ICurColor[i], CV_GRAY2BGR);

        // Now initialize mosaic refinement structure
        refine_mosaic2.Initialize(size_template_x, size_template_y, grid_x, grid_y, offset_templates,
                                  mosaicker.current_warp_color, mosaicker.current_mask,
                                  &Visibility_map);

        // Run mosaic refinement
        refine_mosaic2.Process(mosaicker.connections);

        // Print mosaic
        refine_mosaic2.PrintMosaic("/home/rodrigolinhares/code/bundle_adjustment/code/storage/results_2015_07_17/1_od1_after.png");
//        cv::imshow("Mosaic", refine_mosaic2.Mosaic);
//        key = 0;
//        while(key != 'q')
//            key = cv::waitKey(1);
//
//        cv::destroyAllWindows();
//    }

	return 0;
}
