#ifndef ba_H
#define ba_H

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <omp.h>
#include <fstream>
#include <iomanip>
#include "utils.h"

#define NUM_THREADS 6
#define INACTIVE 0
#define ACTIVE 1

typedef std::pair<cv::Mat, cv::Mat> MatPair;

class ba
{
public:
    void	InitializeMosaic(int grid_x,
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
                             std::string workspace_name = "workspace");

    ~ba();

    void	Process(std::vector<cv::Mat> ICur,
					std::vector<cv::Mat> MaskICur,
                    std::vector<cv::Mat> ICurColor);
	
	void	ProcessVisibilityMap();

	void	LoadWorkspace(cv::Mat *Visibility_map, cv::Mat *Template_set[], cv::Mat *Mask_set[]);

    int     GetNumberElements() {return n_elements;}

//    void    add_noise(cv::Mat ICur[], cv::Mat MaskICur[], cv::Mat ICurColor[]);
	
private:

    // Aie

    void	WarpTPS(cv::Mat ICur, cv::Mat* Mask_roi, cv::Mat& current_warp, cv::Mat* current_mask,
                    cv::Mat ctrl_pts_x_w, cv::Mat ctrl_pts_y_w);

    void    WarpTPSColor(cv::Mat *ICurColor, cv::Mat *current_warp_color);

    void    WarpGrad(cv::Mat input, cv::Mat& gradx, cv::Mat& grady);

	void	DefineCtrlPts();

	void	TPSPrecomputations();

	float	Tps(float r);

    float	Norm(float x, float y);


    // Core

    void    ComputeImageDifference();

    float	BuildMosaicUpdate();

    void	BuildMosaicJacobian();

    void    BuildIndividualMosaicJacobian(int index);

    void    ComputeCombinationMasks();

    void    ComputeGradient();

    void    ComputeBigHessian();

    cv::Mat ComputeHessian(cv::Mat* Jacobian1, cv::Mat *Jacobian2, cv::Mat mask);

    void    BuildOffsetJacobians();

    int     UpdateParams();

    // Illumination stuff

    void    DefineCtrlPtsIllum();

    void	TPSPrecomputationsIllum();

    void    IlluminationCompensation(cv::Mat image, cv::Mat mask, cv::Mat& result);

    void    MultichannelIlluminationCompensation(MatPair images, cv::Mat mask, cv::Mat& result);

    void    MountIlluminationJacobian();

    void	ResetIlluminationParam();

    void    InitIllum();

    void    UpdateIlluminationParam();

    void    ApplyCompensation(cv::Mat& result);

    bool is_anchor(int element_index);

    bool is_inactive(int index);

    int anchors_before(int element_index);

    int inactive_templates_before(int index);

    int inactive_templates();

    void show_vector(cv::Mat vector); //TODO remove

    void show_channels(cv::Mat image); //TODO remove

private:

	// Are we using masks?
	bool	using_masks, using_colors;

	// ba parameters
	int		size_template_x,
			size_template_y,
            size_template,
			n_bins,
			size_bins,
			n_active_pixels,
			n_max_iters,
            interp,
            grid_x,
			grid_y,
			n_elements,
            n_anchors,
            n_combinations,
            n_channels,
            n_active_templates;

	float	epsilon;
	
	// ba core stuff
    std::vector<cv::Mat> dummy_mapx, dummy_mapy;
    cv::Mat gradient, hessian, dummy_hessian;
//	cv::Mat *dummy_mapx,
//            *dummy_mapy,
//            gradient,
//            hessian,
//            dummy_hessian;

    std::vector<cv::Mat> dif,
                         hess_precomp,
                         combination_masks;
	
public:

    std::vector<cv::Mat> ctrl_pts_x_w, ctrl_pts_y_w, gradx, grady, SDw, SDx, current_warp,
                         current_mask, current_warp_color;
    std::vector<std::vector<int> > combinations;
    std::vector<int> anchors, connections, combination_states;
    std::vector<cv::Point> anchor_positions;

private:

	// Mosaic parameters
	int	offset_templates;

	cv::Mat	Visibility_map;

    float ncc_weight, min_overlap;

    long inter_regularization_weight, intra_regularization_weight;

    double total_time;

	// TPS parameters
	int	n_ctrl_pts_x,
		n_ctrl_pts_y,
		n_ctrl_pts,
        ctrl_pts_offset_x,
        ctrl_pts_offset_y;

    int	*ctrl_pts_x,
        *ctrl_pts_y;

	cv::Mat Kinv,
		    MKinv,
            MKinvT,
			Ks,
			Ksw;

	cv::Mat M,
			weights;

	// Misc
	cv::Mat Erosion_mask;

    // Non-rigid illumination compensation
    float *parameters_illum;

    int	n_ctrl_pts_xi,
        n_ctrl_pts_yi,
        n_ctrl_ptsi;

    int	*ctrl_pts_xi,
        *ctrl_pts_yi;

    int ref_intens_value;

//    MatPair images, ctrl_pts_w_i, tps, diff_i, SD_i;
    cv::Mat image, ctrl_pts_w_i, tps, diff_i, SD_i; // For single channel

    cv::Mat MKinvi;

    omp_lock_t* locks;

    // Stats
    std::ofstream ncc_stats, time_stats, ssd_stats;

public:
	
	// ba stats
	int		iters;

	// Current entropy
	float	current_entropy;

    // Comepnsated images
    std::vector<cv::Mat> ICurComp;
    cv::Mat mask;
};

#endif
