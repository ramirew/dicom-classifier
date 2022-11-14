cmake_minimum_required(VERSION 3.5)

project(probandoConLibreriaW LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenMP REQUIRED)

#el path donde se encuentra la libreria dicom-classifier-main es:
#/home/jair/Documents/
#para importar la libreria se tiene que hacer el llamado al punto a
#el punto a, se encuentra donde se ha construido, es decir, el directorio build
#y tambien se tiene que agregar el path donde se encuentran los archivos, por ejemplo en
#el directorio DicomClassifier, o metrics, o Preprocesamiento

add_library(DicomClassifier STATIC IMPORTED)
set_target_properties(DicomClassifier PROPERTIES
    IMPORTED_LOCATION "/home/jair/Documents/dicom-classifier-main/build/libDicomClassifier.a"
    INTERFACE_INCLUDE_DIRECTORIES "/home/jair/Documents/dicom-classifier-main/DicomClassifier"
)

add_library(DicomMetrics STATIC IMPORTED)
set_target_properties(DicomMetrics PROPERTIES
    IMPORTED_LOCATION "/home/jair/Documents/dicom-classifier-main/build/libDicomMetrics.a"
    INTERFACE_INCLUDE_DIRECTORIES "/home/jair/Documents/dicom-classifier-main/metrics"
)

add_library(DicomPreprocesamiento STATIC IMPORTED)
set_target_properties(DicomPreprocesamiento PROPERTIES
    IMPORTED_LOCATION "/home/jair/Documents/dicom-classifier-main/build/libDicomPreprocesamiento.a"
    INTERFACE_INCLUDE_DIRECTORIES "/home/jair/Documents/dicom-classifier-main/Preprocesamiento"
)

add_executable(probandoConLibreriaW main.cpp)
#Aqui se agregan las librerias en este caso con el nombre de la declaracion que se ha echo en el
#archivo CMakeLists.txt, por ejemplo DicomClassifier, DicomMetrics o DicomPreprocesamiento
target_link_libraries(probandoConLibreriaW DicomClassifier DicomMetrics DicomPreprocesamiento -ldcmdata -ldcmimgle -ldcmimage -ldcmjpeg -lpthread OpenMP::OpenMP_CXX)



