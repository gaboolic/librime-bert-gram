cmake_minimum_required(VERSION 3.16)
project(libime-bert-plugin)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required libraries
find_package(PkgConfig REQUIRED)
pkg_check_modules(LIBIME REQUIRED libime-core libime-pinyin)

# Find BERT inference library (example with ONNX Runtime)
# You may need to adjust this based on your BERT implementation
# find_package(onnxruntime REQUIRED)

# Include directories
include_directories(
    ${LIBIME_INCLUDE_DIRS}
)

# Source files
set(SOURCES
    bert_language_model.cpp
)

set(HEADERS
    bert_language_model.h
)

# Create shared library
add_library(libime-bert-plugin SHARED
    ${SOURCES}
    ${HEADERS}
)

target_link_libraries(libime-bert-plugin
    ${LIBIME_LIBRARIES}
    # Add BERT library here, e.g.:
    # onnxruntime::onnxruntime
)

target_include_directories(libime-bert-plugin
    PUBLIC
    ${LIBIME_INCLUDE_DIRS}
)

# Install
install(TARGETS libime-bert-plugin
    LIBRARY DESTINATION lib
)

install(FILES ${HEADERS}
    DESTINATION include/libime-bert-plugin
)

