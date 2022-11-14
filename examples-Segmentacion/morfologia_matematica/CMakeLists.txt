cmake_minimum_required(VERSION 3.5)

project(test_morfologia_matematica LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenMP REQUIRED)

add_library(DicomClassifier STATIC IMPORTED)
set_target_properties(DicomClassifier PROPERTIES
    IMPORTED_LOCATION "/home/debian11/Documentos/dicom-classifier/build2/libDicomClassifier.a"
    INTERFACE_INCLUDE_DIRECTORIES "/home/debian11/Documentos/dicom-classifier/DicomClassifier"
)

add_library(DicomMetrics STATIC IMPORTED)
set_target_properties(DicomMetrics PROPERTIES
    IMPORTED_LOCATION "/home/debian11/Documentos/dicom-classifier/build2/libDicomMetrics.a"
    INTERFACE_INCLUDE_DIRECTORIES "/home/debian11/Documentos/dicom-classifier/metrics"
)

add_library(DicomSegmentacion STATIC IMPORTED)
set_target_properties(DicomSegmentacion PROPERTIES
    IMPORTED_LOCATION "/home/debian11/Documentos/dicom-classifier/build2/libDicomSegmentacion.a"
    INTERFACE_INCLUDE_DIRECTORIES "/home/debian11/Documentos/dicom-classifier/Segmentacion"
)

add_executable(test_morfologia_matematica main.cpp)
target_link_libraries(test_morfologia_matematica DicomClassifier DicomMetrics DicomSegmentacion -ldcmdata -ldcmimgle -ldcmimage -ldcmjpeg -lpthread OpenMP::OpenMP_CXX)
