cmake_minimum_required(VERSION 3.5)

project(Otsu_ LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package( OpenCV REQUIRED)

include_directories( ${OpenCV_INCLUDE_DIRS} )
find_package(OpenMP REQUIRED)

add_library(DicomClassifier STATIC IMPORTED)
set_target_properties(DicomClassifier PROPERTIES
IMPORTED_LOCATION "/home/jbarrionuevop/Desktop/dicom/dicom-classifier/build/libDicomClassifier.a"
INTERFACE_INCLUDE_DIRECTORIES "/home/jbarrionuevop/Desktop/dicom/dicom-classifier/DicomClassifier"
)

add_library(DicomMetrics STATIC IMPORTED)
set_target_properties(DicomMetrics PROPERTIES
    IMPORTED_LOCATION "/home/jbarrionuevop/Desktop/dicom/dicom-classifier/build/libDicomMetrics.a"
    INTERFACE_INCLUDE_DIRECTORIES "/home/jbarrionuevop/Desktop/dicom/dicom-classifier/metrics"
)


add_library(Otsu STATIC IMPORTED)
set_target_properties(Otsu PROPERTIES
    IMPORTED_LOCATION "/home/jbarrionuevop/Desktop/dicom/dicom-classifier/build/libDicomClassifier.a"
    INTERFACE_INCLUDE_DIRECTORIES "/home/jbarrionuevop/Desktop/dicom/dicom-classifier/Otsu"
)

add_executable(Otsu_ main.cpp)

install(TARGETS Otsu_
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

target_link_libraries(Otsu_ DicomClassifier DicomMetrics  Otsu -ldcmdata -ldcmimgle -ldcmimage -ldcmjpeg -lpthread ${OpenCV_LIBS} OpenMP::OpenMP_CXX)
