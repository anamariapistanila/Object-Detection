// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "common.h"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <cmath>
#include <iostream>




void testOpenImage()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		imshow("image",src);
		waitKey();
	}
}

void testOpenImagesFld()
{
	char folderName[MAX_PATH];
	if (openFolderDlg(folderName)==0)
		return;
	char fname[MAX_PATH];
	FileGetter fg(folderName,"bmp");
	while(fg.getNextAbsFile(fname))
	{
		Mat src;
		src = imread(fname);
		imshow(fg.getFoundFileName(),src);
		if (waitKey()==27) //ESC pressed
			break;
	}
}

void testImageOpenAndSave()
{
	Mat src, dst;

	src = imread("Images/Lena_24bits.bmp", CV_LOAD_IMAGE_COLOR);	// Read the image

	if (!src.data)	// Check for invalid input
	{
		printf("Could not open or find the image\n");
		return;
	}

	// Get the image resolution
	Size src_size = Size(src.cols, src.rows);

	// Display window
	const char* WIN_SRC = "Src"; //window for the source image
	namedWindow(WIN_SRC, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Dst"; //window for the destination (processed) image
	namedWindow(WIN_DST, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_DST, src_size.width + 10, 0);

	cvtColor(src, dst, CV_BGR2GRAY); //converts the source image to a grayscale one

	imwrite("Images/Lena_24bits_gray.bmp", dst); //writes the destination to file

	imshow(WIN_SRC, src);
	imshow(WIN_DST, dst);

	printf("Press any key to continue ...\n");
	waitKey(0);
}

void testNegativeImage()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		double t = (double)getTickCount(); // Get the current time [s]
		
		Mat src = imread(fname,CV_LOAD_IMAGE_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height,width,CV_8UC1);
		// Asa se acceseaaza pixelii individuali pt. o imagine cu 8 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i=0; i<height; i++)
		{
			for (int j=0; j<width; j++)
			{
				uchar val = src.at<uchar>(i,j);
				uchar neg = MAX_PATH-val;
				dst.at<uchar>(i,j) = neg;
			}
		}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image",src);
		imshow("negative image",dst);
		waitKey();
	}
}

void testParcurgereSimplaDiblookStyle()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		int w = src.step; // no dword alignment is done !!!
		Mat dst = src.clone();

		double t = (double)getTickCount(); // Get the current time [s]

		// the fastest approach using the “diblook style”
		uchar *lpSrc = src.data;
		uchar *lpDst = dst.data;
		for (int i = 0; i<height; i++)
			for (int j = 0; j < width; j++) {
				uchar val = lpSrc[i*w + j];
				lpDst[i*w + j] = 255 - val;
				/* sau puteti scrie:
				uchar val = lpSrc[i*width + j];
				lpDst[i*width + j] = 255 - val;
				//	w = width pt. imagini cu 8 biti / pixel
				//	w = 3*width pt. imagini cu 24 biti / pixel
				*/
			}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image", src);
		imshow("negative image", dst);
		waitKey();
	}
}

void testColor2Gray()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src = imread(fname);

		int height = src.rows;
		int width = src.cols;

		Mat dst = Mat(height,width,CV_8UC1);
		
		// Asa se acceseaaza pixelii individuali pt. o imagine RGB 24 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i=0; i<height; i++)
		{
			for (int j=0; j<width; j++)
			{
				Vec3b v3 = src.at<Vec3b>(i,j);
				uchar b = v3[0];
				uchar g = v3[1];
				uchar r = v3[2];
				dst.at<uchar>(i,j) = (r+g+b)/3;
			}
		}
		
		imshow("input image",src);
		imshow("gray image",dst);
		waitKey();
	}
}

void GrayscaleToBinar()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname);

		int height = src.rows;
		int width = src.cols;
		int prag = 80;  //orice valoare[0..255]

		Mat dst = Mat(height, width, CV_8UC1);

		// Asa se acceseaaza pixelii individuali pt. o imagine RGB 24 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				Vec3b v3 = src.at<Vec3b>(i, j);
				uchar grey = v3[0];
				if (grey < prag)
				{
					dst.at<uchar>(i, j) = 0;
				}
				else 
				{ 
					dst.at<uchar>(i, j) = 255;
				}
			}
		}

		imshow("input image", src);   //deschide un chenar pe care pune mat-ul 
		//+ numele imaginii
		imshow("binary image", dst);
		waitKey();
	}
}



void testResize()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		Mat dst1,dst2;
		//without interpolation
		resizeImg(src,dst1,320,false);
		//with interpolation
		resizeImg(src,dst2,320,true);
		imshow("input image",src);
		imshow("resized image (without interpolation)",dst1);
		imshow("resized image (with interpolation)",dst2);
		waitKey();
	}
}

void testCanny()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src,dst,gauss;
		src = imread(fname,CV_LOAD_IMAGE_GRAYSCALE);
		int k = 0.4;
		int pH = 50;
		int pL = k*pH;
		GaussianBlur(src, gauss, Size(5, 5), 0.8, 0.8);
		Canny(gauss,dst,pL,pH,3);
		imshow("input image",src);
		imshow("canny",dst);
		waitKey();
	}
}

void testVideoSequence()
{
	VideoCapture cap("Videos/rubic.avi"); // off-line video from file
	//VideoCapture cap(0);	// live video from web cam
	if (!cap.isOpened()) {
		printf("Cannot open video capture device.\n");
		waitKey();
		return;
	}
		
	Mat edges;
	Mat frame;
	char c;

	while (cap.read(frame))
	{
		Mat grayFrame;
		cvtColor(frame, grayFrame, CV_BGR2GRAY);
		Canny(grayFrame,edges,40,100,3);
		imshow("source", frame);
		imshow("gray", grayFrame);
		imshow("edges", edges);
		c = cvWaitKey();  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished\n"); 
			break;  //ESC pressed
		};
	}
}


void testSnap()
{
	VideoCapture cap(0); // open the deafult camera (i.e. the built in web cam)
	if (!cap.isOpened()) // openenig the video device failed
	{
		printf("Cannot open video capture device.\n");
		return;
	}

	Mat frame;
	char numberStr[256];
	char fileName[256];
	
	// video resolution
	Size capS = Size((int)cap.get(CV_CAP_PROP_FRAME_WIDTH),
		(int)cap.get(CV_CAP_PROP_FRAME_HEIGHT));

	// Display window
	const char* WIN_SRC = "Src"; //window for the source frame
	namedWindow(WIN_SRC, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Snapped"; //window for showing the snapped frame
	namedWindow(WIN_DST, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_DST, capS.width + 10, 0);

	char c;
	int frameNum = -1;
	int frameCount = 0;

	for (;;)
	{
		cap >> frame; // get a new frame from camera
		if (frame.empty())
		{
			printf("End of the video file\n");
			break;
		}

		++frameNum;
		
		imshow(WIN_SRC, frame);

		c = cvWaitKey(10);  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished");
			break;  //ESC pressed
		}
		if (c == 115){ //'s' pressed - snapp the image to a file
			frameCount++;
			fileName[0] = NULL;
			sprintf(numberStr, "%d", frameCount);
			strcat(fileName, "Images/A");
			strcat(fileName, numberStr);
			strcat(fileName, ".bmp");
			bool bSuccess = imwrite(fileName, frame);
			if (!bSuccess) 
			{
				printf("Error writing the snapped image\n");
			}
			else
				imshow(WIN_DST, frame);
		}
	}

}



void setLabel(cv::Mat& im, const std::string label, std::vector<cv::Point>& contour)
{
	int fontface = cv::FONT_HERSHEY_SIMPLEX;
	double scale = 0.4;
	int thickness = 1;
	int baseline = 0;

	cv::Size text = cv::getTextSize(label, fontface, scale, thickness, &baseline);
	cv::Rect r = cv::boundingRect(contour);

	cv::Point pt(r.x + ((r.width - text.width) / 2), r.y + ((r.height + text.height) / 2));
	cv::rectangle(im, pt + cv::Point(0, baseline), pt + cv::Point(text.width, -text.height), CV_RGB(255, 255, 255), CV_FILLED);
	cv::putText(im, label, pt, fontface, scale, CV_RGB(0, 0, 0), thickness, 8);
}


Scalar colorObject(Mat src)
{

	cv::Mat gray;
	cv::cvtColor(src, gray, CV_BGR2GRAY);

	
	cv::Mat bw;
	cv::Canny(gray, bw, 0, 50, 5);

	
	std::vector<std::vector<cv::Point> > contours;
	cv::findContours(bw.clone(), contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);


	for (int i = 0; i < contours.size(); i++)
	{
		Rect _boundingRect = boundingRect(contours[i]);
	
		Scalar culoareObiect = mean(src(_boundingRect), gray(_boundingRect));
		if (culoareObiect != Scalar(255,255,255))
		{
			return culoareObiect;
		}
	}
}
int cornerObject(Mat src)
{
	cv::Mat gray;
	cv::cvtColor(src, gray, CV_BGR2GRAY);


	cv::Mat bw;
	std::vector<cv::Point> approx;
	cv::Canny(gray, bw, 0, 50, 5);


	std::vector<std::vector<cv::Point> > contours;
	cv::findContours(bw.clone(), contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
	for (int i = 0; i < contours.size(); i++)
	{
		cv::approxPolyDP(cv::Mat(contours[i]), approx, cv::arcLength(cv::Mat(contours[i]), true) * 0.02, true);
	}
	return approx.size();
}

void contur1()
{
	
	cv::Mat src = cv::imread("Images/Obj.bmp", CV_LOAD_IMAGE_COLOR);
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat obiect = imread(fname);
		const char* Label = "";
		boolean gasit = false;

		//găsirea contururilor este ca găsirea unui obiect alb din fundal negru.
		//asadar trebuie sa facem conversie Grayscale a imaginii initiale
		cv::Mat gray;
		cv::cvtColor(src, gray, CV_BGR2GRAY);

		// Utilizam Canny pentru calculul gradientului imaginii
		//Canny este un algoritm pentru detectarea marginilor într-o imagine care utilizează praguri de histeriză(Această metodă utilizează mai multe praguri pentru a găsi margini. 
		//Se incepe prin a utiliza pragul superior pentru a găsi începutul unei margini. Odată ce avem un punct de pornire, se urmareste traseul marginii prin imagine pixel cu pixel, marcând o margine ori de câte ori suntem deasupra pragului inferior.
		cv::Mat bw;
		cv::Canny(gray, bw, 0, 50, 5);

		// Gasire contur
		std::vector<std::vector<cv::Point> > contours;
		cv::findContours(bw.clone(), contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);


		std::vector<cv::Point> approx;
		cv::Mat dst = src.clone();
		Vec3b color;
		// Se preia culoarea obiectului din imaginea individuala selectata
		Scalar culoareObiect = colorObject(obiect);

		unsigned char B = culoareObiect[0];
		unsigned char G = culoareObiect[1];
		unsigned char R = culoareObiect[2];
		printf("Color(RGB) obiect imagineInidividuala: %d %d %d \n \n", R, G, B);

		int contur = cornerObject(obiect);
		printf("Nr. colturi obiect imagineIndividuala: %d \n \n", contur);



		for (int i = 0; i < contours.size(); i++)
		{
			//Contur aproximat cu precizie, cu perimetrul conturului
			//obținem o succesiune de puncte de contur, arătate de variabila „approx”
			cv::approxPolyDP(cv::Mat(contours[i]), approx, cv::arcLength(cv::Mat(contours[i]), true) * 0.02, true);

			//Funcția boundingRect() este utilizată pentru a desena un dreptunghi aproximativ în jurul imaginii binare. Această funcție este utilizată pentru a evidenția regiunea de interes după obținerea contururilor dintr-o imagine.
			Rect _boundingRect = boundingRect(contours[i]);
			//se calculeaza media fiecărui canal cromatic(average colour)
			Scalar colorC = mean(src(_boundingRect), gray(_boundingRect));

			unsigned char B = colorC[0];
			unsigned char G = colorC[1];
			unsigned char R = colorC[2];
			printf("Color(RGB) obiect imagineMare: %d %d %d \n", R, G, B);

			printf("Nr.colturi obiect imagineMare: %d \n", approx.size());


			if (approx.size() == 3) {
				setLabel(dst, "TRIANGLE", contours[i]);
				Label = "TRIANGLE";

				if (colorC == culoareObiect)
				{
					gasit = true;

					if (gasit == true)
					{
						drawContours(dst, contours, i, Scalar(0, 255, 0), 2);

						printf("Obiect gasit \n");
						waitKey();
					}
					else {
						printf("Nu exista \n");

					}
				}

			}
			else if (approx.size() == 4) {
				setLabel(dst, "RECTANGULAR", contours[i]);
				Label = "RECTANGULAR";

				if (colorC == culoareObiect)
				{
					gasit = true;

					if (gasit == true)
					{
						drawContours(dst, contours, i, Scalar(0, 255, 0), 2);
						printf("Obiect gasit \n");

						waitKey();
					}
					else {
						printf("Nu exista\n");
					}
				}
			}
			else if (approx.size() == 5) {
				setLabel(dst, "PENTA", contours[i]);
				Label = "PENTA";
				if (colorC == culoareObiect) {
					gasit = true;

					if (gasit == true)
					{
						drawContours(dst, contours, i, Scalar(0, 255, 0), 2);
						printf("Obiect gasit \n");

						waitKey();
					}
					else {
						printf("Nu exista \n");
					}
				}
			}
			else if (approx.size() == 6) {
				setLabel(dst, "HEXAGON", contours[i]);
				Label = "HEXAGON";

				if (colorC == culoareObiect)
				{
					gasit = true;

					if (gasit == true)
					{
						drawContours(dst, contours, i, Scalar(0, 255, 0), 2);
						printf("Obiect gasit \n");

						waitKey();
					}
					else {
						printf("Nu exista\n");
					}
				}
			}
			else if (approx.size() == 7) {
				setLabel(dst, "ARROW", contours[i]);
				Label = "arrow";

				if (colorC == culoareObiect) {
					gasit = true;

					if (gasit == true)
					{
						drawContours(dst, contours, i, Scalar(0, 255, 0), 2);
						printf("Obiect gasit \n");

						waitKey();
					}
					else {
						printf("Nu exista \n");
					}
				}
			}
			else if (approx.size() == 10) {
				setLabel(dst, "STAR", contours[i]);
				Label = "Star";
				if (colorC == culoareObiect) {
					gasit = true;

					if (gasit == true)
					{
						drawContours(dst, contours, i, Scalar(0, 255, 0), 2);
						printf("Obiect gasit \n");

						waitKey();
					}
					else {
						printf("Nu exista \n");
					}
				}
			}
			else
			{

				double area = cv::contourArea(contours[i]);
				cv::Rect r = cv::boundingRect(contours[i]);
				int radius = r.width / 2;

				if (std::abs(1 - ((double)r.width / r.height)) <= 0.2 &&
					std::abs(1 - (area / (CV_PI * std::pow(radius, 2)))) <= 0.2)
				{

					setLabel(dst, "CIRCULAR", contours[i]);
					Label = "Circular";

					if (colorC == culoareObiect) {
						gasit = true;
						if (gasit == true) {
							drawContours(dst, contours, i, Scalar(0, 255, 0), 2);
							printf("Obiect gasit \n");
						}
						else {
							printf("Nu exista \n");
						}
					}
				}
				else {

					setLabel(dst, "ANOTHER OBJECT", contours[i]);
					if (colorC == culoareObiect) {
						//Detectie cercuri
						gasit = true;
						if (gasit == true) {
							drawContours(dst, contours, i, Scalar(0, 255, 0), 2);
							printf("Obiect gasit \n");
						}
						else {
							printf("Nu exista \n");
						}
					}
				}
			}

		}

		if (gasit == false) {
			printf("Nu exista");
		}
			cv::imshow("dst", dst);
			cv::waitKey(0);
		}
	
	}
	

int main()
{
	Mat_ <uchar> img = imread("Images/1_Dilate/wdg2ded1_bw.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	
	int op;
	do
	{
		system("cls");
		destroyAllWindows();
		printf("Menu:\n");
		printf(" 1 - Open image\n");
		printf(" 2 - Open BMP images from folder\n");
		printf(" 3 - Image negative - diblook style\n");
		printf(" 4 - BGR->HSV\n");
		printf(" 5 - Resize image\n");
		printf(" 6 - Canny edge detection\n");
		printf(" 7 - Edges in a video sequence\n");
		printf(" 8 - Snap frame from live video\n");
		printf(" 9 - Proiect\n");
	    printf("Option: ");
		scanf("%d",&op);
		switch (op)
		{
			case 1:
				testOpenImage();
				break;
			case 2:
				testOpenImagesFld();
				break;
			case 3:
				testParcurgereSimplaDiblookStyle(); //diblook style
				break;
			case 4:
				testColor2Gray();
				break;
			case 5:
				testResize();
				break;
			case 6:
				testCanny();
				break;
			case 7:
				testVideoSequence();
				break;
			case 8:
				testSnap();
				break;
			case 9:
				//Proiect();
				contur1();
				break;
	

			
		}
	}
	while (op!=0);
	return 0;
}