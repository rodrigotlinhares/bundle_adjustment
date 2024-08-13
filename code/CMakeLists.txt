CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
PROJECT(BundleAdjustment)

if(CMAKE_COMPILER_IS_GNUCXX OR ${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang")
        SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fopenmp -pthread")
endif()

FIND_PACKAGE(OpenCV REQUIRED)
ADD_EXECUTABLE (single_workspace main.cpp
                                 ba.h
#                                 ba_multichannel.cpp
#                                 ba_illum_multichannel.cpp
                                 ba.cpp
                                 ba_illum.cpp
                                 tracking_aux.h
                                 MOSAIC.cpp
                                 MOSAIC.h
                                 tracking_aux.cpp
                                 utils.h
                                 utils.cpp)

ADD_EXECUTABLE (all_workspaces main_all.cpp
                               ba.h
#                               ba_multichannel.cpp
#                               ba_illum_multichannel.cpp
                               ba.cpp
                               ba_illum.cpp
                               tracking_aux.h
                               MOSAIC.cpp
                               MOSAIC.h
                               tracking_aux.cpp
                               utils.h
                               utils.cpp)

TARGET_LINK_LIBRARIES(single_workspace ${OpenCV_LIBS})
TARGET_LINK_LIBRARIES(all_workspaces ${OpenCV_LIBS})
