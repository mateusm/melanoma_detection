//
//  main.cpp
//  melanoma_detection
//
//  Created by Mateus Mesturini Meruvia on 4/28/16.
//  Copyright © 2016 Mateus Mesturini Meruvia. All rights reserved.
//

#include <iostream>
#include <opencv2/opencv.hpp>

#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv/cxcore.h>
#include <opencv/cvaux.h>


#include<stdio.h>
#include<stdlib.h>
#include <string>
#include <sstream>
#include <fstream>
using namespace std;
using namespace cv;


Mat src1, src2;
Mat src_gray;
int thresh = 100;
int max_thresh = 255;
int n = 0;


struct square_roi{
    int x1, y1 = INT_MAX;
    int x2, y2 = INT_MIN;
};

Mat segmentation(Mat image_grayscale){
    Mat segmentated_image;
    
    //erode(image_grayscale, image_grayscale, 10); //NOT WORKING #3 argument = element (?)
    
    threshold(image_grayscale, image_grayscale, 120, 255, THRESH_BINARY_INV);
    
    Mat image_floodfill = image_grayscale.clone();
    floodFill(image_floodfill, cv::Point(0,0), Scalar(255));
    Mat image_floodfill_inv;
    bitwise_not(image_floodfill, image_floodfill_inv);
    Mat image_out = (image_grayscale | image_floodfill_inv);

    return image_out;
}

square_roi ROI(Mat image){
    square_roi roi_image;
    
    for(int y=0;y<image.rows;y++)
    {
        for(int x=0;x<image.cols;x++)
        {
            Vec3b color_image = image.at<Vec3b>(Point(x,y));
            if(color_image[0] == 255 && color_image[1] == 255 && color_image[2] == 255){
                
                if(roi_image.x1 > x)roi_image.x1 = x;
                if(roi_image.y1 > y)roi_image.y1 = y;
                if(roi_image.x2 < x)roi_image.x2 = x;
                if(roi_image.y2 < y)roi_image.y2 = y;
                
            }
        }
    }

    return roi_image;
}

Mat crop(Mat m, Mat out){
    Mat cropped_img;
    vector<vector<Point> > contours;
    vector<Point> points;
    findContours(m, contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);
    for (size_t i=0; i<contours.size(); i++) {
        for (size_t j = 0; j < contours[i].size(); j++) {
            Point p = contours[i][j];
            points.push_back(p);
        }
    }
    
    
    if(points.size() > 0){
        Rect roi = boundingRect(Mat(points).reshape(2));
        //rectangle(out, brect.tl(), brect.br(), Scalar(100, 100, 200), 2, CV_AA);
        
        
        cropped_img = out(roi).clone();
        
    }
    return cropped_img;
}

Mat generate_histogram(Mat src){

    
    /// Separate the image in 3 places ( B, G and R )
    vector<Mat> bgr_planes;
    split( src, bgr_planes );
    
    /// Establish the number of bins
    int histSize = 256;
    
    /// Set the ranges ( for B,G,R) )
    float range[] = { 0, 256 } ;
    const float* histRange = { range };
    
    bool uniform = true; bool accumulate = false;
    
    Mat b_hist, g_hist, r_hist;
    
    /// Compute the histograms:
    calcHist( &bgr_planes[0], 1, 0, Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate );
    calcHist( &bgr_planes[1], 1, 0, Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate );
    calcHist( &bgr_planes[2], 1, 0, Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate );
    
    // Draw the histograms for B, G and R
    int hist_w = 512; int hist_h = 400;
    int bin_w = cvRound( (double) hist_w/histSize );
    
    Mat histImage( hist_h, hist_w, CV_8UC3, Scalar( 0,0,0) );
    
    /// Normalize the result to [ 0, histImage.rows ]
    normalize(b_hist, b_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
    normalize(g_hist, g_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
    normalize(r_hist, r_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
    
    /// Draw for each channel
    for( int i = 1; i < histSize; i++ )
    {
        line( histImage, Point( bin_w*(i-1), hist_h - cvRound(b_hist.at<float>(i-1)) ) ,
             Point( bin_w*(i), hist_h - cvRound(b_hist.at<float>(i)) ),
             Scalar( 255, 0, 0), 2, 8, 0  );
        line( histImage, Point( bin_w*(i-1), hist_h - cvRound(g_hist.at<float>(i-1)) ) ,
             Point( bin_w*(i), hist_h - cvRound(g_hist.at<float>(i)) ),
             Scalar( 0, 255, 0), 2, 8, 0  );
        line( histImage, Point( bin_w*(i-1), hist_h - cvRound(r_hist.at<float>(i-1)) ) ,
             Point( bin_w*(i), hist_h - cvRound(r_hist.at<float>(i)) ),
             Scalar( 0, 0, 255), 2, 8, 0  );
    }
    
    return histImage;
}

double Comp_Hist(Mat src_base, Mat src_test, int compare_method){
    Mat hsv_base;
    Mat hsv_test;
    

    
    /// Convert to HSV
    cvtColor( src_base, hsv_base, COLOR_BGR2HSV );
    cvtColor( src_test, hsv_test, COLOR_BGR2HSV );

    
    
    /// Using 50 bins for hue and 60 for saturation
    int h_bins = 50; int s_bins = 60;
    int histSize[] = { h_bins, s_bins };
    
    // hue varies from 0 to 179, saturation from 0 to 255
    float h_ranges[] = { 0, 180 };
    float s_ranges[] = { 0, 256 };
    
    const float* ranges[] = { h_ranges, s_ranges };
    
    // Use the o-th and 1-st channels
    int channels[] = { 0, 1 };
    
    
    /// Histograms
    MatND hist_base;
    MatND hist_test;
    
    /// Calculate the histograms for the HSV images
    calcHist( &hsv_base, 1, channels, Mat(), hist_base, 2, histSize, ranges, true, false );
    normalize( hist_base, hist_base, 0, 1, NORM_MINMAX, -1, Mat() );
    
    calcHist( &hsv_test, 1, channels, Mat(), hist_test, 2, histSize, ranges, true, false );
    normalize( hist_test, hist_test, 0, 1, NORM_MINMAX, -1, Mat() );
    
    
    /// Apply the histogram comparison methods
    /*
    for( int i = 0; i < 4; i++ )
    {
        int compare_method = i;
        double base_base = compareHist( hist_base, hist_base, compare_method );
        double base_test = compareHist( hist_base, hist_test, compare_method );

        

        switch (i) {
            case(0): printf( " CORRELATION \n Base-Base: %f \n Base-Test(1): %f \n\n", base_base, base_test); return base_test;
            case(1): printf( " CHI-SQUARE \n Base-Base: %f \n Base-Test(1): %f \n\n", base_base, base_test); return base_test;
            case(2): printf( " INTERSECTION \n Base-Base: %f \n Base-Test(1): %f \n\n", base_base, base_test); return base_test;
            case(3): printf( " BHATTACHARYYA \n Base-Base: %f \n Base-Test(1): %f \n\n", base_base, base_test); return base_test;
            
        }
        
        
        
    }*/
    
    double base_test = compareHist( hist_base, hist_test, compare_method );
    switch (compare_method) {
        case(0): printf( " CORRELATION \n Base-Test(1): %f \n\n", base_test); return base_test;
        case(1): printf( " CHI-SQUARE \n Base-Test(1): %f \n\n", base_test); return base_test;
        case(2): printf( " INTERSECTION \n Base-Test(1): %f \n\n", base_test); return base_test;
        case(3): printf( " BHATTACHARYYA \n Base-Test(1): %f \n\n", base_test); return base_test;
    }
    printf( "Done \n" );
    return 0;
    
}

Mat read_img(int index, int type){
    Mat image;
    string source;
    string prefix = "/Users/mateusmesturini/Dropbox/Morgan Classes/Senior Project/Set2/data/img";
    source = prefix + to_string(index) + ".jpg";
    
    switch (type) {
        case(0):    image = imread(source); break;
        case(1):    image = imread(source, IMREAD_GRAYSCALE); break;
    }
    
    return image;
}


class Image
{
private:
    Mat file;
    Mat file_gray;
public:
    
    Image(string source){
        file = imread(source);
        file_gray = imread(source, IMREAD_GRAYSCALE);
    }
    
    



};








int main(int argc, const char * argv[]) {
    
    //METHODS:
    // 0 -> CORRELATION
    // 1 -> CHI-SQUARE
    // 2 -> INTERSECTION
    // 3 -> BHATTACHARYYA
    int method = 0;
    
    int src_index = 1;
    Mat src = read_img(src_index, 0);
    Mat src_gray = read_img(src_index, 1);
    src_gray = segmentation(src_gray);
    src = crop(src_gray, src);


    double match_higher = 0;
    double match_lower = DBL_MAX;

    int best_match = 0;
    
    for(int i = 1; i <=40 ; i++){
        Mat test = read_img(i, 0);
        Mat test_gray = read_img(i, 1);
        test_gray = segmentation(test_gray);
        test = crop(test_gray, test);
        printf("%d\n",i);
        
        
        
        if(method%2 == 0){
        
            if (Comp_Hist(src, test,method) > match_higher && Comp_Hist(src, test,method) != 1) {
                match_higher = Comp_Hist(src, test,method);
                best_match = i;
            }
        }else{
            if (Comp_Hist(src, test,method) < match_lower && src_index!=i) {
                match_lower = Comp_Hist(src, test,method);
                best_match = i;
        
        }
        }
        
        
        
        
    } //Compares the image with the others
    
    printf("%d\n", best_match);
    Mat img_match = read_img(best_match, 0);
    Mat img_match_gray = read_img(best_match, 1);
    img_match_gray = segmentation(img_match_gray);
    img_match = crop(img_match_gray, img_match);
    
    namedWindow("src", CV_WINDOW_AUTOSIZE);
    namedWindow("best_match", CV_WINDOW_AUTOSIZE);
    imshow("src", src);
    imshow("best_match", img_match);

        

    
    waitKey(0);
    
        //imwrite(destination_graythresh[i].c_str(), im_th);
    return 0;
}