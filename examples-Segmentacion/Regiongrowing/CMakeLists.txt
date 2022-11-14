cmake_minimum_required(VERSION 3.5)

project(Regiongrowing VERSION 0.1 LANGUAGES CXX)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(QT NAMES Qt6 Qt5 REQUIRED COMPONENTS Widgets)
find_package(Qt${QT_VERSION_MAJOR} REQUIRED COMPONENTS Widgets)

set(PROJECT_SOURCES
        main.cpp
        regiongrowing.cpp
        regiongrowing.h
        regiongrowing.ui
)

if(${QT_VERSION_MAJOR} GREATER_EQUAL 6)
    qt_add_executable(Regiongrowing
        MANUAL_FINALIZATION
        ${PROJECT_SOURCES}
    )
# Define target properties for Android with Qt 6 as:
#    set_property(TARGET Regiongrowing APPEND PROPERTY QT_ANDROID_PACKAGE_SOURCE_DIR
#                 ${CMAKE_CURRENT_SOURCE_DIR}/android)
# For more information, see https://doc.qt.io/qt-6/qt-add-executable.html#target-creation
else()
    if(ANDROID)
        add_library(Regiongrowing SHARED
            ${PROJECT_SOURCES}
        )
# Define properties for Android with Qt 5 after find_package() calls as:
#    set(ANDROID_PACKAGE_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/android")
    else()
        add_executable(Regiongrowing
            ${PROJECT_SOURCES}
        )
    endif()
endif()

target_link_libraries(Regiongrowing PRIVATE Qt${QT_VERSION_MAJOR}::Widgets)

set_target_properties(Regiongrowing PROPERTIES
    MACOSX_BUNDLE_GUI_IDENTIFIER my.example.com
    MACOSX_BUNDLE_BUNDLE_VERSION ${PROJECT_VERSION}
    MACOSX_BUNDLE_SHORT_VERSION_STRING ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}
    MACOSX_BUNDLE TRUE
    WIN32_EXECUTABLE TRUE
)

install(TARGETS Regiongrowing
    BUNDLE DESTINATION .
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

if(QT_VERSION_MAJOR EQUAL 6)
    qt_finalize_executable(Regiongrowing)
endif()
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

# Define headers
file(GLOB_RECURSE HEADERS_FILES
    "*.h"
    "*.hpp"
)

# Set sources
add_library(DicomClassifier STATIC "${SOURCES_FILES}")
add_library(DicomMetrics STATIC "${METRICS_SOURCES_FILES}")

# Include dirs
include_directories(DicomClassifier)
include_directories(metrics)


# C++ flags
set(USE_PTHREAD ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")

