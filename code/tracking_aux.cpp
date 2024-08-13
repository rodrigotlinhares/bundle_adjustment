#include "tracking_aux.h"


// Gets video
void	OnMouse(int event,
				int x,
				int y,
				int,
				void* test)
{
	int *mouse_coords = (int*) test;

	if(event == CV_EVENT_LBUTTONDOWN)
	{
		mouse_coords[0] = x;
		mouse_coords[1] = y;

		printf("Coords selected!\n");
	}    
}

//char	nbgetchar()
//{
//	char key;
//	if(key = _kbhit())
//		key = _getch();

//	return key;
//}

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
					   int *interp)
{
    cv::FileStorage fs(path, cv::FileStorage::READ);

	*grid_x = fs["grid_x"];
	*grid_y = fs["grid_y"];
	*offset_templates = fs["offset_templates"];
	*size_template_x = fs["size_template_x"];
	*size_template_y = fs["size_template_y"];
	*size_bins = fs["size_bins"];
	*epsilon = fs["epsilon"];
	*n_bins = fs["n_bins"];
	*n_max_iters = fs["n_max_iters"];
	*confidence_thres = fs["confidence_thres"];
	*percentage_active = fs["percentage_active"];
    *filename = (std::string) fs["filename"];
    *interp = fs["interp"];
    *n_ctrl_pts_x = fs["n_ctrl_pts_x"];
    *n_ctrl_pts_y = fs["n_ctrl_pts_y"];
    *n_ctrl_pts_xi = fs["n_ctrl_pts_x_i"];
    *n_ctrl_pts_yi = fs["n_ctrl_pts_y_i"];
	*lambda = fs["lambda_tps"];
}
