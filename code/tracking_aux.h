#include <stdio.h>
//#include <conio.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>


// Tracking aux fcts

void	OnMouse(int event, int x, int y, int, void* test);

char	nbgetchar();

void	LoadParameters(std::string path,
                       int *grid_x,
                       int *grid_y,
                       int *offset_templates,
                       int *size_template_x,
                       int *size_template_y,
                       int *n_ctrl_pts_x,
                       int *n_ctrl_pts_y,
                       int *n_ctrl_pts_xi,
                       int *n_ctrl_pts_yi,
                       float *lambda,
                       int *size_bins,
                       float *epsilon,
                       int *n_bins,
                       int *n_max_iters,
                       float *confidence_thres,
                       int *percentage_active,
                       std::string *filename,
                       int *interp);

void	DefineTemplate(cv::Mat *Template,
					   cv::Mat *ICur,
					   int *coords,
					   int isgrayscale);
