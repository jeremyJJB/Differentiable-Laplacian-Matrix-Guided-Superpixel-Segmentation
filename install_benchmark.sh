git clone --recursive https://github.com/davidstutz/superpixel-benchmark.git

cd superpixel-benchmark

# Below are the fixes for using cv2 4.0, the code as is uses 2.4

# lib_eval/evaluation.cpp
sed -i \
-e '524s/CV_DIST_L2/cv::DIST_L2/' \
-e '722s/CV_BGR2GRAY/cv::COLOR_BGR2GRAY/' \
-e '758s/CV_BGR2GRAY/cv::COLOR_BGR2GRAY/' \
lib_eval/evaluation.cpp

# lib_eval/evaluation_summary.cpp
sed -i \
-e '1191s/CV_LOAD_IMAGE_COLOR/cv::IMREAD_COLOR/' \
lib_eval/evaluation_summary.cpp

# lib_eval/transformation.cpp
sed -i \
-e '81s/CV_BGR2GRAY/cv::COLOR_BGR2GRAY/' \
lib_eval/transformation.cpp

# lib_eval/robustness_tool.cpp
sed -i \
-e '515s/CV_BGR2Lab/cv::COLOR_BGR2Lab/' \
-e '520s/CV_Lab2BGR/cv::COLOR_Lab2BGR/' \
lib_eval/robustness_tool.cpp

# lib_etps/spixel.cpp
sed -i \
-e '78s/CV_LOAD_IMAGE_COLOR/cv::IMREAD_COLOR/' \
-e '124s/CV_LOAD_IMAGE_COLOR/cv::IMREAD_COLOR/' \
-e '125s/CV_LOAD_IMAGE_ANYDEPTH/cv::IMREAD_ANYDEPTH/' \
-e '174s/CV_LOAD_IMAGE_COLOR/cv::IMREAD_COLOR/' \
-e '175s/CV_LOAD_IMAGE_COLOR/cv::IMREAD_COLOR/' \
lib_etps/spixel.cpp

# lib_crs/crs_opencv.h
sed -i \
-e '87s/CV_BGR2YCrCb/cv::COLOR_BGR2YCrCb/' \
lib_crs/crs_opencv.h


sed -i '1i add_definitions(-DBOOST_TIMER_ENABLE_DEPRECATED)' CMakeLists.txt
cmake .
make
