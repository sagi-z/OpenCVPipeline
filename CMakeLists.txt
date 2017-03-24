# cmake needs this line
cmake_minimum_required(VERSION 2.8)

add_definitions(-std=c++11 -Werror=return-type)

# Define project name
project(smiledetect_tbb)

# Find OpenCV, you may need to set OpenCV_DIR variable
# to the absolute path to the directory containing OpenCVConfig.cmake file
# via the command line or GUI
find_package(OpenCV REQUIRED)

# If the package has been found, several variables will
# be set, you can find the full list with descriptions
# in the OpenCVConfig.cmake file.
# Print some message showing some of them
message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

if(CMAKE_VERSION VERSION_LESS "2.8.11")
  # Add OpenCV headers location to your include paths
  include_directories(${OpenCV_INCLUDE_DIRS})
endif()

find_package(Threads REQUIRED)

find_package(PkgConfig)
if (PKG_CONFIG_FOUND)
  pkg_check_modules(TBB "tbb")
  if (TBB_FOUND)
    add_definitions(${TBB_CFLAGS} ${TBB_CFLAGS_OTHER})
    MESSAGE(tbb version : ${TBB_VERSION})
    include_directories(
        ${TBB_INCLUDE_DIRS}
    )
  endif()
endif()

# Declare the executable target built from your sources
add_executable(smiledetect smiledetect.cpp)
add_executable(smiledetect_tbb smiledetect_tbb.cpp)

# Link your application with OpenCV libraries
target_link_libraries(smiledetect ${OpenCV_LIBS})
target_link_libraries(smiledetect_tbb ${OpenCV_LIBS} ${TBB_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})
