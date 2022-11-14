cmake_minimum_required(VERSION 3.5)

project(canny_libreria LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenMP REQUIRED)

add_library(DicomClassifier STATIC IMPORTED)
set_target_properties(DicomClassifier PROPERTIES
    IMPORTED_LOCATION "/home/usuario/Documentos/libreria/dicom-classifier-main/build/libDicomClassifier.a"
    INTERFACE_INCLUDE_DIRECTORIES "/home/usuario/Documentos/libreria/dicom-classifier-main/DicomClassifier"
)

add_library(DicomMetrics STATIC IMPORTED)
set_target_properties(DicomMetrics PROPERTIES
    IMPORTED_LOCATION "/home/usuario/Documentos/libreria/dicom-classifier-main/build/libDicomMetrics.a"
    INTERFACE_INCLUDE_DIRECTORIES "/home/usuario/Documentos/libreria/dicom-classifier-main/metrics"
)

add_library(DicomSegmentacion STATIC IMPORTED)
set_target_properties(DicomSegmentacion PROPERTIES
    IMPORTED_LOCATION "/home/usuario/Documentos/libreria/dicom-classifier-main/build/libDicomSegmentacion.a"
    INTERFACE_INCLUDE_DIRECTORIES "/home/usuario/Documentos/libreria/dicom-classifier-main/Segmentacion"
)

add_executable(canny_libreria main.cpp)
target_link_libraries(canny_libreria DicomClassifier DicomMetrics DicomSegmentacion -ldcmdata -ldcmimgle -ldcmimage -ldcmjpeg -lpthread OpenMP::OpenMP_CXX)
