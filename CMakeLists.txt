cmake_minimum_required(VERSION 3.0.0)

project(DicomClassifier VERSION 0.1.0)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Define sources
file(GLOB_RECURSE SOURCES_FILES
    "DicomClassifier/*.cpp"
    "DicomClassifier/*.cxx"
)

file(GLOB_RECURSE METRICS_SOURCES_FILES
    "metrics/*.cpp"
)

#Añadiendo ruta de nuestro algoritmo
file(GLOB_RECURSE SEGMENTACION_SOURCES_FILES
    "Segmentacion/*.cpp"
)

file(GLOB_RECURSE Preprocesamiento_SOURCES_FILES
    "Preprocesamiento/*.cpp"
)

# Define headers
file(GLOB_RECURSE HEADERS_FILES
    "*.h"
    "*.hpp"
)

# Set sources
add_library(DicomClassifier STATIC "${SOURCES_FILES}")
add_library(DicomMetrics STATIC "${METRICS_SOURCES_FILES}")
# Añadiendo algoritmo a la librería
add_library(DicomSegmentacion STATIC "${SEGMENTACION_SOURCES_FILES}")
add_library(DicomPreprocesamiento STATIC "${Preprocesamiento_SOURCES_FILES}")

# Include dirs
include_directories(DicomClassifier)
include_directories(metrics)
# Añadiendo directorio a la librería
include_directories(DicomSegmentacion)
include_directories(DicomPreprocesamiento)

# C++ flags
set(USE_PTHREAD ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")

# AÑADIR LIBRERÍAS DE OPENMP
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()
