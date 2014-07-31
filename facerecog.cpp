
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

/** Function Headers */
void detectAndDisplay( Mat frame );
void calcHist( Mat frame );
void histEql( Mat frame );


/** Global variables */
String face_cascade_name = "lbpcascade_frontalface.xml";
String eyes_cascade_name = "haarcascade_eye_tree_eyeglasses.xml";
CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;
string window_name = "Capture - Face detection";

RNG rng(12345);
//Face dimensions and attributes
int height, width, xpos, ypos;
std::vector<Rect> faces;
Mat frame_gray;

/**
 * @function main
 */
int main( void )
{
  VideoCapture capture;
  Mat frame, reframe;

  //-- 1. Load the cascade
  if( !face_cascade.load( face_cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };
  if( !eyes_cascade.load( eyes_cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };

  //-- 2. Read the video stream
  capture.open( -1 );
  if( capture.isOpened() )
  {
    for(;;)
    {
        capture >> frame;
		//-- 3. Apply the classifier to the frame
        if( !frame.empty() )
        {
			detectAndDisplay( frame ); 
			//-- Show what you got
            //imshow( window_name, frame );
            
            
            //-- Calculating Histogram 
			calcHist( frame );
			
			//-- Histogram equalization
            histEql( frame );
            
            //-- Resize to use in recognition
            resize(frame, reframe, reframe.size(), 0.5, 0.5,INTER_CUBIC );
            imshow( "resized image ", reframe );

            
        }
        else
		{ 
			printf(" --(!) No captured frame -- Break!"); break; }
		int c = waitKey(10);
        if( (char)c == 'c' ) { break; }
	
     }
  }
  return 0;
}

/**
 * @function 
 */
void detectAndDisplay( Mat frame )
{
   cvtColor( frame, frame_gray, COLOR_BGR2GRAY );
   equalizeHist( frame_gray, frame_gray );
	//-- Detect faces
	
   face_cascade.detectMultiScale( frame_gray, faces, 1.1, 2, 0, Size(80, 80) );
   for( size_t i = 0; i < faces.size(); i++ )
    {
		if ( faces[i].x >320 )
		cout << "move left " << endl;
		else 
		cout << "move right " << endl;
		
		/*if (faces[i].height > 120 )
		cout << "zoom out " << endl;
		else if (faces[i].height < 50)
		cout << "zoom in " << endl;
		*/
	  //cout << faces[i].width << "  x  " << faces[i].height << endl;
	  
      Mat faceROI = frame_gray( faces[i] );
      std::vector<Rect> eyes;
	  //-- In each face, detect eyes
      eyes_cascade.detectMultiScale( faceROI, eyes, 1.1, 2, 0 |CV_HAAR_SCALE_IMAGE, Size(30, 30) );
      if( eyes.size() == 2)
      {
         //-- Draw the face
         Point center( faces[i].x + faces[i].width/2, faces[i].y + faces[i].height/2 );
         //ellipse( frame, center, Size( faces[i].width/2, faces[i].height/2), 0, 0, 360, Scalar( 255, 0, 0 ), 2, 8, 0 );
		  rectangle( frame, faces[i] ,CV_RGB(0, 0,255), 3);
         for( size_t j = 0; j < eyes.size(); j++ )
          { //-- Draw the eyes
            Point eye_center( faces[i].x + eyes[j].x + eyes[j].width/2, faces[i].y + eyes[j].y + eyes[j].height/2 );
            int radius = cvRound( (eyes[j].width + eyes[j].height)*0.25 );
            circle( frame, eye_center, radius, Scalar( 255, 0, 0 ), 3, 8, 0 );
          }
       }
	 } 
}

void calcHist( Mat frame)
{
	/// Separate the image in 3 places ( B, G and R )
	vector<Mat> bgr_planes;
	split( frame, bgr_planes );

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
	/// Display
	//namedWindow("calcHist Demo", WINDOW_AUTOSIZE );
	//imshow("calcHist Demo", histImage );
            
}

void histEql( Mat frame )
{
	vector<Mat> channels; 
	Mat img_hist_equalized;
	cvtColor(frame, img_hist_equalized, CV_BGR2YCrCb); //change the color image from BGR to YCrCb format	
	split(img_hist_equalized,channels); //split the image into channels
	equalizeHist(channels[0], channels[0]); //equalize histogram on the 1st channel (Y)
	merge(channels,img_hist_equalized); //merge 3 channels including the modified 1st channel into one image
	cvtColor(img_hist_equalized, img_hist_equalized, CV_YCrCb2BGR); //change the color image from YCrCb to BGR format (to display image properly)
    //imshow("eqlHist", img_hist_equalized );     
}   
	
