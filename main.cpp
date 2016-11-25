#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include <cmath>
#include <math.h> 
#include <vector>
#include <algorithm>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "highgui.h"
#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

    Mat ellipse5 = (Mat_<uchar>(5,5) << 0, 0, 1, 0, 0,
                                         1, 1, 1, 1, 1,
                                         1, 1, 1, 1, 1,
                                         1, 1, 1, 1, 1,
                                         0, 0, 1, 0, 0 );

    Mat ellipse3 = (Mat_<uchar>(3,3) << 0, 1, 0,
                                         1, 1, 1,
                                         0, 1, 0 );
    
    Mat kernel = (Mat_<float>(3,3) <<   1,  1, 1,
                                        1, -8, 1,
                                        1,  1, 1);

const int FILTER_WINDOW = 5;
const int HIST_DIM = 2;
const int BLUR_WINDOW = 3;
const int PARAM_H = 45;
const int TEMP_WINDOW = 7;
const int SEARCH_WINDOW = 21;
const int MORPH_ITER = 2;

float thresholdPeri = 0.25;


int main(int argc, char const *argv[])
{
    Mat img, tmp;

    if(argc<3)
    {
        cout<< "Usage: ./main template.jpg source.jpg"<<endl; 
        return -1;
    }
    
    if( argc != 3 || !(tmp=imread(argv[1], 1)).data || !(img=imread(argv[2], 1)).data )
    {
        cout<<"Image not present!"<<endl;
        return -1;
    }

    // Read template and source image
    Mat templateImg,source;

    // Resizing the image
    resize( tmp, templateImg, Size( tmp.cols*0.9, tmp.rows*0.9 ));
    resize( img, source, Size( img.cols*0.9, img.rows*0.9 ));

    // Convert to HSV
    Mat templateImg_hsv,source_hsv;
    cvtColor(templateImg, templateImg_hsv, CV_BGR2HSV);
    cvtColor(source, source_hsv, CV_BGR2HSV);

    // Noise removal
    medianBlur( templateImg_hsv, templateImg_hsv,FILTER_WINDOW);
    medianBlur( source_hsv, source_hsv, FILTER_WINDOW);
 
    // Spilt template into H S V channels
    vector<Mat> templateImg_hsvPlanes;
    split(templateImg_hsv.clone(), templateImg_hsvPlanes);

    // H --(0 to 179)-- S --(0 to 255)--  
    float hranges[] = { 0, 180};
    float sranges[] = { 0, 256};

    // Histogram bins
    int hbins = 180, sbins = 256;
    int histSize[] = {hbins, sbins};
    const float* ranges[] = { hranges, sranges };
    
    // computing Histogram for H and S
    int channels[] = {0, 1};

    Mat templateImg_hist, source_hist;

    // Compute Histogram
    calcHist( &templateImg_hsv, 1, channels, Mat(), // do not use mask //Number of images =1;
             templateImg_hist, HIST_DIM, histSize, ranges,
             true, // the histogram is uniform
             false );

    calcHist( &source_hsv, 1, channels, Mat(), // do not use mask
             source_hist, HIST_DIM, histSize, ranges,
             true, // the histogram is uniform
             false );


    // New matrix for back projection
   Mat backproj = Mat::zeros(source.rows,source.cols,CV_8UC(1));
     
     // <---------------- Histogram Back Projection Algo ---------------->
    for (int row = 0; row < source.rows; row++)
    {
        for (int col = 0; col < source.cols; col++)
        {
            Scalar Intensity_img = source_hsv.at<Vec3b>(row,col);
            
            float freq_temp = cvRound(templateImg_hist.at<float>(Intensity_img[0],Intensity_img[1]));
            float freq_img = cvRound(source_hist.at<float>(Intensity_img[0],Intensity_img[1]));
            float ratio = freq_temp/(freq_img+1);

            ratio = (ratio>1)?1:ratio;
            
            backproj.at<uchar>(row,col) = 255*(ratio);
            
        }
    }
    // <------------------------------------------------------------------->

    //imshow("back",backproj);
    
    // Normalize backprojected image (0 to 255)
    normalize(backproj,backproj,0,255,NORM_MINMAX,-1,Mat());
    medianBlur( backproj, backproj, BLUR_WINDOW);
    
    // Denoising image
    Mat denoise;
    fastNlMeansDenoising(backproj,denoise, PARAM_H, TEMP_WINDOW, SEARCH_WINDOW); 
    
    // Thresholding (GrayScale to Binary)
    Mat backprojThresh;
    threshold(denoise.clone(),backprojThresh,0,255,CV_THRESH_BINARY|CV_THRESH_OTSU);


    // Finding distance transform
    Mat dist;
    distanceTransform(backprojThresh, dist, CV_DIST_L2, 3);
    
    // Normalize the distance image for range = {0.0, 1.0}
    normalize(dist, dist, 0, 1., NORM_MINMAX);
    threshold(dist, dist, .1, 1., CV_THRESH_BINARY);

    // Dilate a bit the dist image
    Mat kernel1 = Mat::ones(5, 5, CV_8UC1);
    dilate(dist, dist, kernel1);

    // Converting to 8UC single channel
    Mat dist_8u;
    dist.convertTo(dist_8u, CV_8UC(1));

    // Fill Holes present in contours
    backprojThresh = dist_8u;
    Mat im_floodfill = backprojThresh.clone();
    floodFill(im_floodfill, cv::Point(0,0), Scalar(255));
    Mat im_floodfill_inv;
    bitwise_not(im_floodfill, im_floodfill_inv);
     
    // Combine the two images to get the complete object
    backprojThresh = (backprojThresh | im_floodfill_inv);
    
    // Morphological operation OPEN --> First Erosion then Dilation, Using Kernel ellipse (5,5)
    morphologyEx(backprojThresh,backprojThresh,MORPH_OPEN,ellipse5,Point(-1,-1), MORPH_ITER); //white increase
 
    // Finding contours
    vector<Vec4i> hierarchy;
    vector<vector<Point> > contours;
    findContours( backprojThresh.clone(), contours, hierarchy, CV_RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    
    // Filtering contours based on Perimeter of contours. Obscure flowers are removed here
    // Setting threshold Perimeter to be 35% of Maximum perimeter
    float maxPerimeter = 0.0;

        for( int z = 0; z < contours.size(); z++ )
        {
            double peri = arcLength(contours[z],true);
            //cout<<peri<<endl;
            if(maxPerimeter<peri)
            {
                maxPerimeter = peri;
            }
        }
        int counter=0;

        for( int x = 0; x < contours.size(); x++ )
        {
            double p = arcLength(contours[x],true)/maxPerimeter;
            if(p>thresholdPeri)
                counter++;
        }


    cout<< "Number of flowers: " << counter<<endl;

    imshow("Source Image", img);
    imshow("Template Image", tmp);

    waitKey(0);
}


