////
// Compare with smiledetect.cpp sample provided with opencv
////
#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>

using namespace std;
using namespace cv;

static void help()
{
    cout << "\nThis program demonstrates the smile detector.\n"
            "Usage:\n"
            "./smiledetect_tbb [--cascade=<cascade_path> this is the frontal face classifier]\n"
            "   [--smile-cascade=[<smile_cascade_path>]]\n"
            "   [--scale=<image scale greater or equal to 1, try 2.0 for example. The larger the faster the processing>]\n"
            "   [--try-flip]\n"
            "   [video_filename|camera_index]\n\n"
            "Example:\n"
            "./smiledetect_tbb --cascade=\"../../data/haarcascades/haarcascade_frontalface_alt.xml\" --smile-cascade=\"../../data/haarcascades/haarcascade_smile.xml\" --scale=2.0\n\n"
            "During execution:\n\tHit any key to quit.\n"
            "\tUsing OpenCV version " << CV_VERSION << "\n" << endl;
}

// TBB NOTE: we need these headers
#include <thread>
#include <tbb/concurrent_queue.h>
#include <tbb/pipeline.h>
volatile bool done = false; // volatile is enough here. We don't need a mutex for this simple flag.
struct ProcessingChainData
{
    Mat img;
    vector<Rect> faces, faces2;
    Mat gray, smallImg;
};
void detectAndDrawTBB( VideoCapture &capture,
                       tbb::concurrent_bounded_queue<ProcessingChainData *> &guiQueue,
                       CascadeClassifier& cascade,
                       CascadeClassifier& nestedCascade,
                       double scale, bool tryflip );

string cascadeName;
string nestedCascadeName;

int main( int argc, const char** argv )
{
    VideoCapture capture;
    Mat frame, image;
    string inputName;
    bool tryflip;

    help();

    // TBB NOTE: these are not thread safe, so be careful not to use them in parallel.
    CascadeClassifier cascade, nestedCascade;
    double scale;
    cv::CommandLineParser parser(argc, argv,
        "{help h||}{scale|1|}"
        "{cascade|../../data/haarcascades/haarcascade_frontalface_alt.xml|}"
        "{smile-cascade|../../data/haarcascades/haarcascade_smile.xml|}"
        "{try-flip||}{@input||}");
    if (parser.has("help"))
    {
        help();
        return 0;
    }
    cascadeName = parser.get<string>("cascade");
    nestedCascadeName = parser.get<string>("smile-cascade");
    tryflip = parser.has("try-flip");
    inputName = parser.get<string>("@input");
    scale = parser.get<int>("scale");
    if (!parser.check())
    {
        help();
        return 1;
    }
    if (scale < 1)
        scale = 1;
    if( !cascade.load( cascadeName ) )
    {
        cerr << "ERROR: Could not load face cascade" << endl;
        help();
        return -1;
    }
    if( !nestedCascade.load( nestedCascadeName ) )
    {
        cerr << "ERROR: Could not load smile cascade" << endl;
        help();
        return -1;
    }
    if( inputName.empty() || (isdigit(inputName[0]) && inputName.size() == 1) )
    {
        int c = inputName.empty() ? 0 : inputName[0] - '0' ;
        if(!capture.open(c))
            cout << "Capture from camera #" <<  c << " didn't work" << endl;
    }
    else if( inputName.size() )
    {
        if(!capture.open( inputName ))
            cout << "Could not read " << inputName << endl;
    }

    int64 startTime;
    if( capture.isOpened() )
    {
        cout << "Video capturing has been started ..." << endl;
        cout << endl << "NOTE: Smile intensity will only be valid after a first smile has been detected" << endl;

        tbb::concurrent_bounded_queue<ProcessingChainData *> guiQueue;
        guiQueue.set_capacity(2); // TBB NOTE: flow control so the pipeline won't fill too much RAM
        auto pipelineRunner = thread( detectAndDrawTBB, ref(capture), ref(guiQueue), ref(cascade), ref(nestedCascade), scale, tryflip );

        startTime = getTickCount();

        // TBB NOTE: GUI is executed in main thread
        ProcessingChainData *pData=0;
        for(;! done;)
        {
            if (guiQueue.try_pop(pData))
            {
                char c = (char)waitKey(1);
                if( c == 27 || c == 'q' || c == 'Q' )
                {
                    done = true;
                }
                imshow( "result", pData->img );
                delete pData;
                pData = 0;
            }
        }
        double tfreq = getTickFrequency();
        double secs = ((double) getTickCount() - startTime)/tfreq;
        cout << "Execution took " << fixed << secs << " seconds." << endl;
        // TBB NOTE: flush the queue after marking 'done'
        do
        {
            delete pData;
        } while (guiQueue.try_pop(pData));
        pipelineRunner.join(); // TBB NOTE: wait for the pipeline to finish
    }
    else
    {
        cerr << "ERROR: Could not initiate capture" << endl;
        help();
        return -1;
    }

    return 0;
}

// TBB NOTE: This usage below is just for the tbb demonstration.
//           It is not an example for good OO code. The lambda
//           expressions are used to easily show the correleation
//           between the original code and the tbb code.
void detectAndDrawTBB( VideoCapture &capture,
                       tbb::concurrent_bounded_queue<ProcessingChainData *> &guiQueue,
                       CascadeClassifier& cascade,
                       CascadeClassifier& nestedCascade,
                       double scale, bool tryflip )
{
    const static Scalar colors[] =
    {
        Scalar(255,0,0),
        Scalar(255,128,0),
        Scalar(255,255,0),
        Scalar(0,255,0),
        Scalar(0,128,255),
        Scalar(0,255,255),
        Scalar(0,0,255),
        Scalar(255,0,255)
    };

    tbb::parallel_pipeline(7, // TBB NOTE: (recommendation) NumberOfFilters
                           // 1st filter
                           tbb::make_filter<void,ProcessingChainData*>(tbb::filter::serial_in_order,
                                                                  [&](tbb::flow_control& fc)->ProcessingChainData*
                          {   // TBB NOTE: this filter feeds input into the pipeline
                              Mat frame;
                              capture >> frame;
                              if( done || frame.empty() )
                              {
                                  // 'done' is our own exit flag
                                  // being set and checked in and out
                                  // of the pipeline
                                  done = true;

                                  // These 2 lines are how to tell TBB to stop the pipeline
                                  fc.stop();
                                  return 0;
                              }
                              auto pData = new ProcessingChainData;
                              pData->img = frame.clone();
                              return pData;
                          }
                          )&
                           // 2nd filter
                           tbb::make_filter<ProcessingChainData*,ProcessingChainData*>(tbb::filter::serial_in_order,
                                                                  [&](ProcessingChainData *pData)->ProcessingChainData*
                          {
                              cvtColor( pData->img, pData->gray, COLOR_BGR2GRAY );
                              return pData;
                          }
                          )&
                           // 3rd filter
                           tbb::make_filter<ProcessingChainData*,ProcessingChainData*>(tbb::filter::serial_in_order,
                                                                  [&](ProcessingChainData *pData)->ProcessingChainData*
                          {
                              double fx = 1 / scale;
                              resize( pData->gray, pData->smallImg, Size(), fx, fx, INTER_LINEAR );
                              return pData;
                          }
                          )&
                           // 4th filter
                           tbb::make_filter<ProcessingChainData*,ProcessingChainData*>(tbb::filter::serial_in_order,
                                                                  [&](ProcessingChainData *pData)->ProcessingChainData*
                          {
                              equalizeHist( pData->smallImg, pData->smallImg );
                              return pData;
                          }
                          )&
                           // 5th filter
                           tbb::make_filter<ProcessingChainData*,ProcessingChainData*>(tbb::filter::serial_in_order,
                                                                  [&](ProcessingChainData *pData)->ProcessingChainData*
                          {
                              cascade.detectMultiScale( pData->smallImg, pData->faces,
                                                        1.1, 2, 0
                                                        //|CASCADE_FIND_BIGGEST_OBJECT
                                                        //|CASCADE_DO_ROUGH_SEARCH
                                                        |CASCADE_SCALE_IMAGE,
                                                        Size(30, 30) );
                              if( tryflip )
                              {   // TBB NOTE: 1. CascadeClassifier is already paralleled by OpenCV
                                  //           2. Is is not thread safe, so don't call the same classifier from different threads.
                                  flip(pData->smallImg, pData->smallImg, 1);
                                  cascade.detectMultiScale( pData->smallImg, pData->faces2,
                                                            1.1, 2, 0
                                                            //|CASCADE_FIND_BIGGEST_OBJECT
                                                            //|CASCADE_DO_ROUGH_SEARCH
                                                            |CASCADE_SCALE_IMAGE,
                                                            Size(30, 30) );
                                  for( vector<Rect>::const_iterator r = pData->faces2.begin(); r != pData->faces2.end(); ++r )
                                  {
                                      pData->faces.push_back(Rect(pData->smallImg.cols - r->x - r->width, r->y, r->width, r->height));
                                  }
                              }
                              return pData;
                          }
                          )&
                           // 6th filter
                           tbb::make_filter<ProcessingChainData*,ProcessingChainData*>(tbb::filter::serial_in_order,
                                                                  [&](ProcessingChainData *pData)->ProcessingChainData*
                          {
                              for ( size_t i = 0; i < pData->faces.size(); i++ )
                              {
                                  Rect r = pData->faces[i];
                                  Mat smallImgROI;
                                  vector<Rect> nestedObjects;
                                  Point center;
                                  Scalar color = colors[i%8];
                                  int radius;

                                  double aspect_ratio = (double)r.width/r.height;
                                  if( 0.75 < aspect_ratio && aspect_ratio < 1.3 )
                                  {
                                      center.x = cvRound((r.x + r.width*0.5)*scale);
                                      center.y = cvRound((r.y + r.height*0.5)*scale);
                                      radius = cvRound((r.width + r.height)*0.25*scale);
                                      circle( pData->img, center, radius, color, 3, 8, 0 );
                                  }
                                  else
                                      rectangle( pData->img, cvPoint(cvRound(r.x*scale), cvRound(r.y*scale)),
                                                 cvPoint(cvRound((r.x + r.width-1)*scale), cvRound((r.y + r.height-1)*scale)),
                                                 color, 3, 8, 0);

                                  const int half_height=cvRound((float)r.height/2);
                                  r.y=r.y + half_height;
                                  r.height = half_height-1;
                                  smallImgROI = pData->smallImg( r );
                                  nestedCascade.detectMultiScale( smallImgROI, nestedObjects,
                                                                  1.1, 0, 0
                                                                  //|CASCADE_FIND_BIGGEST_OBJECT
                                                                  //|CASCADE_DO_ROUGH_SEARCH
                                                                  //|CASCADE_DO_CANNY_PRUNING
                                                                  |CASCADE_SCALE_IMAGE,
                                                                  Size(30, 30) );

                                  // The number of detected neighbors depends on image size (and also illumination, etc.). The
                                  // following steps use a floating minimum and maximum of neighbors. Intensity thus estimated will be
                                  //accurate only after a first smile has been displayed by the user.
                                  const int smile_neighbors = (int)nestedObjects.size();
                                  static int max_neighbors=-1;
                                  static int min_neighbors=-1;
                                  if (min_neighbors == -1) min_neighbors = smile_neighbors;
                                  max_neighbors = MAX(max_neighbors, smile_neighbors);

                                  // Draw rectangle on the left side of the image reflecting smile intensity
                                  float intensityZeroOne = ((float)smile_neighbors - min_neighbors) / (max_neighbors - min_neighbors + 1);
                                  int rect_height = cvRound((float)pData->img.rows * intensityZeroOne);
                                  Scalar col = Scalar((float)255 * intensityZeroOne, 0, 0);
                                  rectangle(pData->img, cvPoint(0, pData->img.rows), cvPoint(pData->img.cols/10, pData->img.rows - rect_height), col, -1);
                              }
                              return pData;
                          }
                          )&
                           // 7th filter
                           tbb::make_filter<ProcessingChainData*,void>(tbb::filter::serial_in_order,
                                                                  [&](ProcessingChainData *pData)
                          {   // TBB NOTE: pipeline end point. dispatch to GUI
                              if (! done)
                              {
                                  try
                                  {
                                      guiQueue.push(pData);
                                  }
                                  catch (...)
                                  {
                                      cout << "Pipeline caught an exception on the queue" << endl;
                                      done = true;
                                  }
                              }
                          }
                          )
                          );

}
