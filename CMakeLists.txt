cmake_minimum_required(VERSION 3.10)
project(rime-bert-grammar)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set runtime library to match glog (static runtime /MT)
# glog is built with /MT (static runtime), so we need to match it
# This must be set BEFORE any targets are created
if(MSVC)
  # Use CMAKE_MSVC_RUNTIME_LIBRARY if available (CMake 3.15+)
  if(POLICY CMP0091)
    cmake_policy(SET CMP0091 NEW)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
  endif()
  # Also force replace /MD with /MT in all flags (for compatibility)
  foreach(flag_var CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_RELEASE CMAKE_CXX_FLAGS_DEBUG
                   CMAKE_C_FLAGS CMAKE_C_FLAGS_RELEASE CMAKE_C_FLAGS_DEBUG)
    if(${flag_var})
      string(REPLACE "/MD" "/MT" ${flag_var} "${${flag_var}}")
      string(REPLACE "/MDd" "/MTd" ${flag_var} "${${flag_var}}")
    endif()
  endforeach()
  # Set default for new targets
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MT")
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MTd")
  set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} /MT")
  set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} /MTd")
endif()

# Collect source files, excluding test files
file(GLOB bert_grammar_src 
  "src/*.cc"
  "src/*.cpp"
)
# Remove test files from the list
list(FILTER bert_grammar_src EXCLUDE REGEX ".*test.*\\.(cc|cpp)$")

# Option to enable/disable ONNX Runtime
option(ENABLE_ONNXRUNTIME "Enable ONNX Runtime support" ON)

# Option to specify Rime build directory (for build_config.h)
set(RIME_BUILD_DIR_VAR "" CACHE PATH "Path to librime build directory (contains rime/build_config.h)")

set(ONNXRUNTIME_FOUND FALSE)
set(ONNXRUNTIME_LIBRARIES "")
set(ONNXRUNTIME_INCLUDE_DIRS "")

if(ENABLE_ONNXRUNTIME)
  # Try to find ONNX Runtime
  # Method 1: Use find_package if available
  find_package(onnxruntime QUIET)
  
  if(onnxruntime_FOUND)
    set(ONNXRUNTIME_FOUND TRUE)
    message(STATUS "Found ONNX Runtime via find_package")
  else()
    # Method 2: Manual path specification
    # Set these variables before configuring, or uncomment and set paths:
    # set(ONNXRUNTIME_ROOT_DIR "/path/to/onnxruntime")
    
    if(DEFINED ONNXRUNTIME_ROOT_DIR)
      set(ONNXRUNTIME_INCLUDE_DIRS "${ONNXRUNTIME_ROOT_DIR}/include")
      
      # Find the library file
      if(WIN32)
        find_library(ONNXRUNTIME_LIB
          NAMES onnxruntime
          PATHS "${ONNXRUNTIME_ROOT_DIR}/lib"
          NO_DEFAULT_PATH
        )
      else()
        find_library(ONNXRUNTIME_LIB
          NAMES onnxruntime
          PATHS "${ONNXRUNTIME_ROOT_DIR}/lib"
          NO_DEFAULT_PATH
        )
      endif()
      
      if(ONNXRUNTIME_LIB)
        set(ONNXRUNTIME_LIBRARIES ${ONNXRUNTIME_LIB})
        set(ONNXRUNTIME_FOUND TRUE)
        message(STATUS "Found ONNX Runtime at ${ONNXRUNTIME_ROOT_DIR}")
      endif()
    endif()
  endif()
  
  if(ONNXRUNTIME_FOUND)
    add_definitions(-DRIME_USE_ONNXRUNTIME)
    message(STATUS "ONNX Runtime: ENABLED")
    message(STATUS "  Include dirs: ${ONNXRUNTIME_INCLUDE_DIRS}")
    message(STATUS "  Libraries: ${ONNXRUNTIME_LIBRARIES}")
  else()
    message(WARNING "ONNX Runtime not found. BERT Grammar will not work.")
    message(STATUS "To enable ONNX Runtime:")
    message(STATUS "  1. Download from https://github.com/microsoft/onnxruntime/releases")
    message(STATUS "  2. Set ONNXRUNTIME_ROOT_DIR to the installation directory")
    message(STATUS "  3. Or install via package manager and use find_package")
  endif()
else()
  message(STATUS "ONNX Runtime: DISABLED")
endif()

# Find Boost library (required by librime)
# First check if Boost is in librime's deps directory
if(NOT DEFINED BOOST_ROOT AND DEFINED RIME_ROOT_DIR)
  # Check if Boost is in librime's deps directory
  if(EXISTS "${RIME_ROOT_DIR}/deps/boost-1.89.0")
    set(BOOST_ROOT "${RIME_ROOT_DIR}/deps/boost-1.89.0")
    message(STATUS "Found Boost in librime deps: ${BOOST_ROOT}")
  elseif(EXISTS "${RIME_ROOT_DIR}/deps/boost")
    set(BOOST_ROOT "${RIME_ROOT_DIR}/deps/boost")
    message(STATUS "Found Boost in librime deps: ${BOOST_ROOT}")
  endif()
endif()

find_package(Boost QUIET COMPONENTS signals2 filesystem system)
if(Boost_FOUND)
  message(STATUS "Found Boost: ${Boost_VERSION}")
  message(STATUS "  Include dirs: ${Boost_INCLUDE_DIRS}")
  message(STATUS "  Libraries: ${Boost_LIBRARIES}")
else()
  # Try to find Boost manually
  find_path(BOOST_INCLUDE_DIR boost/signals2/connection.hpp
    PATHS
      "${BOOST_ROOT}/include"
      "${BOOST_ROOT}"
      "${RIME_ROOT_DIR}/deps/boost-1.89.0"
      "${RIME_ROOT_DIR}/deps/boost"
      "C:/local/boost"
      "C:/boost"
      "/usr/local/include"
      "/usr/include"
  )
  
  if(BOOST_INCLUDE_DIR)
    set(Boost_FOUND TRUE)
    set(Boost_INCLUDE_DIRS ${BOOST_INCLUDE_DIR})
    message(STATUS "Found Boost manually at ${BOOST_INCLUDE_DIR}")
  else()
    message(WARNING "Boost not found. Please install Boost or set BOOST_ROOT.")
    message(STATUS "  Windows: Download from https://www.boost.org/ or use vcpkg: vcpkg install boost")
    message(STATUS "  Linux: sudo apt-get install libboost-dev")
    message(STATUS "  macOS: brew install boost")
    if(DEFINED RIME_ROOT_DIR)
      message(STATUS "  Or check: ${RIME_ROOT_DIR}/deps/boost-1.89.0")
    endif()
  endif()
endif()

# Find glog library (required by both plugin and test)
# Option to specify glog root directory
set(GLOG_ROOT "" CACHE PATH "Path to glog root directory (contains include/glog/logging.h)")

# First check if glog is in librime's deps directory
if(NOT GLOG_ROOT AND DEFINED RIME_ROOT_DIR)
  # Check common glog directory names in deps
  set(GLOG_DEPS_PATHS
    "${RIME_ROOT_DIR}/deps/glog"
    "${RIME_ROOT_DIR}/deps/google-glog"
  )
  
  foreach(GLOG_DEPS_PATH ${GLOG_DEPS_PATHS})
    # Check for installed structure (include/glog/logging.h)
    if(EXISTS "${GLOG_DEPS_PATH}/include/glog/logging.h")
      set(GLOG_ROOT "${GLOG_DEPS_PATH}")
      set(GLOG_INCLUDE_SUBDIR "include")
      message(STATUS "Found glog in librime deps (installed): ${GLOG_ROOT}")
      break()
    # Check for source structure (src/glog/logging.h)
    elseif(EXISTS "${GLOG_DEPS_PATH}/src/glog/logging.h")
      set(GLOG_ROOT "${GLOG_DEPS_PATH}")
      set(GLOG_INCLUDE_SUBDIR "src")
      message(STATUS "Found glog in librime deps (source): ${GLOG_ROOT}")
      break()
    endif()
  endforeach()
  
  # Also check for any glog-* directories
  if(NOT GLOG_ROOT)
    file(GLOB GLOG_DEPS_DIRS "${RIME_ROOT_DIR}/deps/glog-*")
    foreach(GLOG_DIR ${GLOG_DEPS_DIRS})
      # Check installed structure
      if(IS_DIRECTORY "${GLOG_DIR}" AND EXISTS "${GLOG_DIR}/include/glog/logging.h")
        set(GLOG_ROOT "${GLOG_DIR}")
        set(GLOG_INCLUDE_SUBDIR "include")
        message(STATUS "Found glog in librime deps (installed): ${GLOG_ROOT}")
        break()
      # Check source structure
      elseif(IS_DIRECTORY "${GLOG_DIR}" AND EXISTS "${GLOG_DIR}/src/glog/logging.h")
        set(GLOG_ROOT "${GLOG_DIR}")
        set(GLOG_INCLUDE_SUBDIR "src")
        message(STATUS "Found glog in librime deps (source): ${GLOG_ROOT}")
        break()
      endif()
    endforeach()
  endif()
  
  # Also check in build directory (if librime was built)
  if(NOT GLOG_ROOT AND EXISTS "${RIME_ROOT_DIR}/build")
    file(GLOB GLOG_BUILD_DIRS 
      "${RIME_ROOT_DIR}/build/deps/glog*"
      "${RIME_ROOT_DIR}/build/third_party/glog*"
      "${RIME_ROOT_DIR}/build/_deps/glog*"
    )
    foreach(GLOG_DIR ${GLOG_BUILD_DIRS})
      if(IS_DIRECTORY "${GLOG_DIR}" AND EXISTS "${GLOG_DIR}/include/glog/logging.h")
        set(GLOG_ROOT "${GLOG_DIR}")
        message(STATUS "Found glog in librime build: ${GLOG_ROOT}")
        break()
      endif()
    endforeach()
  endif()
endif()

find_package(glog QUIET)
if(NOT glog_FOUND)
  # Try to find glog manually
  find_path(GLOG_INCLUDE_DIR glog/logging.h
    PATHS
      "${GLOG_ROOT}/${GLOG_INCLUDE_SUBDIR}"
      "${GLOG_ROOT}/include"
      "${GLOG_ROOT}/src"
      "${GLOG_ROOT}"
      "${RIME_ROOT_DIR}/deps/glog/src"
      "${RIME_ROOT_DIR}/deps/glog/include"
      "${RIME_ROOT_DIR}/deps/google-glog/include"
      "${RIME_ROOT_DIR}/deps/google-glog/src"
      "${RIME_ROOT_DIR}/build/deps/glog/include"
      "${RIME_ROOT_DIR}/build/deps/glog/src"
      "${RIME_ROOT_DIR}/build/deps/google-glog/include"
      "${RIME_ROOT_DIR}/build/third_party/glog/include"
      "${RIME_ROOT_DIR}/third_party/glog/include"
      "C:/local/glog/include"
      "C:/glog/include"
      "/usr/local/include"
      "/usr/include"
    NO_DEFAULT_PATH
  )
  
  # If not found, try with default paths
  if(NOT GLOG_INCLUDE_DIR)
    find_path(GLOG_INCLUDE_DIR glog/logging.h)
  endif()
  
  find_library(GLOG_LIBRARY
    NAMES glog glogd
    PATHS
      "${GLOG_ROOT}/lib"
      "${GLOG_ROOT}/lib/Release"
      "${GLOG_ROOT}/lib/Debug"
      "${RIME_ROOT_DIR}/deps/glog/lib"
      "${RIME_ROOT_DIR}/deps/google-glog/lib"
      "${RIME_ROOT_DIR}/build/deps/glog/lib"
      "${RIME_ROOT_DIR}/build/deps/google-glog/lib"
      "${RIME_ROOT_DIR}/build/third_party/glog/lib"
      "${RIME_ROOT_DIR}/third_party/glog/lib"
      "C:/local/glog/lib"
      "C:/glog/lib"
      "/usr/local/lib"
      "/usr/lib"
    NO_DEFAULT_PATH
  )
  
  # If not found, try with default paths
  if(NOT GLOG_LIBRARY)
    find_library(GLOG_LIBRARY NAMES glog glogd)
  endif()
  
  if(GLOG_INCLUDE_DIR)
    set(glog_FOUND TRUE)
    message(STATUS "Found glog manually")
    message(STATUS "  Include dir: ${GLOG_INCLUDE_DIR}")
    if(GLOG_LIBRARY)
      message(STATUS "  Library: ${GLOG_LIBRARY}")
    else()
      message(STATUS "  Library: not found (header-only or source build)")
    endif()
  else()
    message(WARNING "glog not found. Please install glog or set GLOG_ROOT.")
    message(STATUS "  Windows: Download from https://github.com/google/glog or use vcpkg: vcpkg install glog")
    message(STATUS "  Linux: sudo apt-get install libglog-dev")
    message(STATUS "  macOS: brew install glog")
    if(DEFINED RIME_ROOT_DIR)
      message(STATUS "  Or check: ${RIME_ROOT_DIR}/deps/glog*")
    endif()
  endif()
endif()

# Find Rime library (needed for both plugin and test)
set(RIME_FOUND FALSE)
set(RIME_LIBRARIES "")
set(RIME_INCLUDE_DIRS "")

# Method 1: Try find_package (works on Windows and Unix)
find_package(rime QUIET)
if(rime_FOUND)
  set(RIME_FOUND TRUE)
  # Use variables from find_package
  if(TARGET rime::rime)
    set(RIME_LIBRARIES rime::rime)
  elseif(rime_LIBRARIES)
    set(RIME_LIBRARIES ${rime_LIBRARIES})
  endif()
  if(rime_INCLUDE_DIRS)
    set(RIME_INCLUDE_DIRS ${rime_INCLUDE_DIRS})
  endif()
  message(STATUS "Found Rime via find_package")
else()
  # Method 2: Try PkgConfig (Unix only)
  if(UNIX AND NOT WIN32)
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
      pkg_check_modules(RIME_PKG QUIET rime)
      if(RIME_PKG_FOUND)
        set(RIME_FOUND TRUE)
        set(RIME_LIBRARIES ${RIME_PKG_LIBRARIES})
        set(RIME_INCLUDE_DIRS ${RIME_PKG_INCLUDE_DIRS})
        message(STATUS "Found Rime via PkgConfig")
      endif()
    endif()
  endif()
endif()

# Method 3: Manual specification (if RIME_ROOT_DIR is set)
if(NOT RIME_FOUND AND DEFINED RIME_ROOT_DIR)
  # Check if it's a source directory (has src/rime/) or installed directory (has include/rime/)
  set(RIME_BUILD_DIR "")
  if(EXISTS "${RIME_ROOT_DIR}/src/rime/common.h")
    # Source directory
    set(RIME_INCLUDE_DIRS "${RIME_ROOT_DIR}/src")
    message(STATUS "Detected Rime source directory")
    
    # Try to find build directory with build_config.h
    # Common build directory names
    set(BUILD_DIR_NAMES "build" "out" "build/Release" "build/Debug" "cmake-build-release" "cmake-build-debug")
    foreach(BUILD_DIR_NAME ${BUILD_DIR_NAMES})
      set(BUILD_DIR "${RIME_ROOT_DIR}/${BUILD_DIR_NAME}")
      if(EXISTS "${BUILD_DIR}/rime/build_config.h")
        set(RIME_BUILD_DIR "${BUILD_DIR}")
        message(STATUS "Found Rime build directory at ${RIME_BUILD_DIR}")
        break()
      endif()
    endforeach()
    
    # Also check if RIME_BUILD_DIR is explicitly set via CMake variable
    if(NOT RIME_BUILD_DIR AND RIME_BUILD_DIR_VAR)
      if(EXISTS "${RIME_BUILD_DIR_VAR}/rime/build_config.h")
        set(RIME_BUILD_DIR "${RIME_BUILD_DIR_VAR}")
        message(STATUS "Using explicitly set Rime build directory: ${RIME_BUILD_DIR}")
      else()
        message(WARNING "RIME_BUILD_DIR set but build_config.h not found at ${RIME_BUILD_DIR_VAR}/rime/build_config.h")
      endif()
    endif()
    
    # If build_config.h is not found, try to generate it from build_config.h.in
    if(NOT RIME_BUILD_DIR)
      # Look for build_config.h.in in the source directory
      set(BUILD_CONFIG_IN "")
      if(EXISTS "${RIME_ROOT_DIR}/src/rime/build_config.h.in")
        set(BUILD_CONFIG_IN "${RIME_ROOT_DIR}/src/rime/build_config.h.in")
      elseif(EXISTS "${RIME_ROOT_DIR}/rime/build_config.h.in")
        set(BUILD_CONFIG_IN "${RIME_ROOT_DIR}/rime/build_config.h.in")
      else()
        # Search for it
        file(GLOB_RECURSE BUILD_CONFIG_IN_CANDIDATES "${RIME_ROOT_DIR}/*/build_config.h.in")
        if(BUILD_CONFIG_IN_CANDIDATES)
          list(GET BUILD_CONFIG_IN_CANDIDATES 0 BUILD_CONFIG_IN)
        endif()
      endif()
      
      if(BUILD_CONFIG_IN)
        message(STATUS "Found build_config.h.in at ${BUILD_CONFIG_IN}, generating build_config.h")
        # Create output directory
        set(BUILD_CONFIG_OUT "${CMAKE_BINARY_DIR}/rime/build_config.h")
        file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/rime")
        
        # Read the template
        file(READ "${BUILD_CONFIG_IN}" TEMPLATE_CONTENT)
        
        # Replace CMake variables with minimal values
        string(REPLACE "@RIME_DATA_DIR@" "\"\"" TEMPLATE_CONTENT "${TEMPLATE_CONTENT}")
        string(REPLACE "@RIME_PLUGINS_DIR@" "\"\"" TEMPLATE_CONTENT "${TEMPLATE_CONTENT}")
        string(REPLACE "#cmakedefine RIME_ENABLE_LOGGING" "// #define RIME_ENABLE_LOGGING" TEMPLATE_CONTENT "${TEMPLATE_CONTENT}")
        string(REPLACE "#cmakedefine RIME_ALSO_LOG_TO_STDERR" "// #define RIME_ALSO_LOG_TO_STDERR" TEMPLATE_CONTENT "${TEMPLATE_CONTENT}")
        string(REPLACE "#cmakedefine RIME_DATA_DIR" "// #define RIME_DATA_DIR" TEMPLATE_CONTENT "${TEMPLATE_CONTENT}")
        string(REPLACE "#cmakedefine RIME_PLUGINS_DIR" "// #define RIME_PLUGINS_DIR" TEMPLATE_CONTENT "${TEMPLATE_CONTENT}")
        
        # Add header comment
        set(OUTPUT_CONTENT "// build_config.h generated from build_config.h.in by rime-bert-grammar\n")
        set(OUTPUT_CONTENT "${OUTPUT_CONTENT}// This is a minimal version. For full functionality, build librime first.\n")
        set(OUTPUT_CONTENT "${OUTPUT_CONTENT}//\n")
        set(OUTPUT_CONTENT "${OUTPUT_CONTENT}${TEMPLATE_CONTENT}")
        
        # Write the generated file
        file(WRITE "${BUILD_CONFIG_OUT}" "${OUTPUT_CONTENT}")
        set(RIME_BUILD_DIR "${CMAKE_BINARY_DIR}")
        message(STATUS "Generated build_config.h at ${BUILD_CONFIG_OUT}")
      else()
        # Fallback: create a minimal one
        message(WARNING "build_config.h not found and build_config.h.in not found. Creating a minimal build_config.h")
        message(STATUS "  For full functionality, build librime first: cd librime && mkdir build && cd build && cmake .. && cmake --build .")
        
        set(MINIMAL_BUILD_CONFIG_H "${CMAKE_BINARY_DIR}/rime/build_config.h")
        file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/rime")
        file(WRITE "${MINIMAL_BUILD_CONFIG_H}" 
"// Minimal build_config.h generated by rime-bert-grammar CMake
// For full functionality, build librime first to generate the complete build_config.h

#ifndef RIME_BUILD_CONFIG_H_
#define RIME_BUILD_CONFIG_H_

// Minimal definitions - may need to be adjusted based on your librime version

#endif  // RIME_BUILD_CONFIG_H_
")
        set(RIME_BUILD_DIR "${CMAKE_BINARY_DIR}")
        message(STATUS "Created minimal build_config.h at ${MINIMAL_BUILD_CONFIG_H}")
      endif()
    endif()
    
  elseif(EXISTS "${RIME_ROOT_DIR}/include/rime/common.h")
    # Installed directory
    set(RIME_INCLUDE_DIRS "${RIME_ROOT_DIR}/include")
    message(STATUS "Detected Rime installed directory")
  else()
    # Try both
    if(EXISTS "${RIME_ROOT_DIR}/src")
      set(RIME_INCLUDE_DIRS "${RIME_ROOT_DIR}/src")
    else()
      set(RIME_INCLUDE_DIRS "${RIME_ROOT_DIR}/include")
    endif()
  endif()
  
  # Verify the include directory is correct
  if(EXISTS "${RIME_INCLUDE_DIRS}/rime/common.h")
    set(RIME_FOUND TRUE)
    message(STATUS "Found Rime headers at ${RIME_INCLUDE_DIRS}")
    
    # Add build directory to include paths if found
    if(RIME_BUILD_DIR)
      list(APPEND RIME_INCLUDE_DIRS "${RIME_BUILD_DIR}")
    endif()
    
    # Try to find library (optional for source builds)
    if(WIN32)
      find_library(RIME_LIB
        NAMES rime
        PATHS "${RIME_ROOT_DIR}/lib" "${RIME_ROOT_DIR}/lib/Release" "${RIME_ROOT_DIR}/build" "${RIME_ROOT_DIR}/build/Release"
        NO_DEFAULT_PATH
      )
    else()
      find_library(RIME_LIB
        NAMES rime
        PATHS "${RIME_ROOT_DIR}/lib" "${RIME_ROOT_DIR}/build"
        NO_DEFAULT_PATH
      )
    endif()
    if(RIME_LIB)
      set(RIME_LIBRARIES ${RIME_LIB})
      message(STATUS "Found Rime library at ${RIME_LIB}")
    else()
      message(STATUS "Rime library not found, but headers are available (may be source-only build)")
    endif()
  else()
    set(RIME_FOUND FALSE)
    message(WARNING "Could not find rime/common.h in ${RIME_INCLUDE_DIRS}")
  endif()
endif()

if(NOT RIME_FOUND)
  message(WARNING "Rime library not found. Please set RIME_ROOT_DIR or ensure find_package(rime) works.")
  message(STATUS "Example: cmake .. -DRIME_ROOT_DIR=/path/to/rime")
else()
  message(STATUS "Rime: Found")
  message(STATUS "  Include dirs: ${RIME_INCLUDE_DIRS}")
  message(STATUS "  Libraries: ${RIME_LIBRARIES}")
endif()

add_library(rime-bert-grammar-objs OBJECT ${bert_grammar_src})
if(BUILD_SHARED_LIBS)
  set_target_properties(rime-bert-grammar-objs
    PROPERTIES
    POSITION_INDEPENDENT_CODE ON)
endif()

# Set runtime library for object library to match glog (static /MT)
if(MSVC)
  if(POLICY CMP0091)
    set_target_properties(rime-bert-grammar-objs PROPERTIES
      MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
  else()
    set_target_properties(rime-bert-grammar-objs PROPERTIES
      COMPILE_FLAGS_RELEASE "/MT"
      COMPILE_FLAGS_DEBUG "/MTd")
  endif()
endif()

# Add include directories
if(RIME_FOUND AND RIME_INCLUDE_DIRS)
  target_include_directories(rime-bert-grammar-objs PRIVATE
    ${RIME_INCLUDE_DIRS})
endif()

if(Boost_FOUND AND Boost_INCLUDE_DIRS)
  target_include_directories(rime-bert-grammar-objs PRIVATE
    ${Boost_INCLUDE_DIRS})
endif()

if(ONNXRUNTIME_FOUND)
  target_include_directories(rime-bert-grammar-objs PRIVATE
    ${ONNXRUNTIME_INCLUDE_DIRS})
endif()

# Add src directory for headers
target_include_directories(rime-bert-grammar-objs PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src)

# Add glog include directory and definitions
if(glog_FOUND AND GLOG_INCLUDE_DIR)
  target_include_directories(rime-bert-grammar-objs PRIVATE
    ${GLOG_INCLUDE_DIR})
  # Add glog compile definitions for newer versions
  target_compile_definitions(rime-bert-grammar-objs PRIVATE
    GLOG_NO_ABBREVIATED_SEVERITIES
)
elseif(TARGET glog::glog)
  # Use target if available (handled by target_link_libraries)
elseif(GLOG_INCLUDE_DIR)
  # Even if glog_FOUND is not set, if we found the include dir, use it
  target_include_directories(rime-bert-grammar-objs PRIVATE
    ${GLOG_INCLUDE_DIR})
  # Add glog compile definitions for newer versions
  target_compile_definitions(rime-bert-grammar-objs PRIVATE
    GLOG_NO_ABBREVIATED_SEVERITIES
)
  message(STATUS "Adding glog include directory: ${GLOG_INCLUDE_DIR}")
endif()

# Link libraries
if(Boost_FOUND AND Boost_LIBRARIES)
  target_link_libraries(rime-bert-grammar-objs
    ${Boost_LIBRARIES})
endif()

if(glog_FOUND)
  if(GLOG_LIBRARY)
    target_link_libraries(rime-bert-grammar-objs
      ${GLOG_LIBRARY})
  elseif(TARGET glog::glog)
    target_link_libraries(rime-bert-grammar-objs
      glog::glog)
  endif()
endif()

if(ONNXRUNTIME_FOUND)
  target_link_libraries(rime-bert-grammar-objs
    ${ONNXRUNTIME_LIBRARIES})
endif()

set(plugin_name rime-bert-grammar PARENT_SCOPE)
set(plugin_objs $<TARGET_OBJECTS:rime-bert-grammar-objs> PARENT_SCOPE)
set(plugin_deps ${rime_library} PARENT_SCOPE)
if(ONNXRUNTIME_FOUND)
  list(APPEND plugin_deps ${ONNXRUNTIME_LIBRARIES})
endif()
set(plugin_deps ${plugin_deps} PARENT_SCOPE)
set(plugin_modules "bert_grammar" PARENT_SCOPE)

# Optional: Build test program
option(BUILD_TEST "Build test program" OFF)

if(BUILD_TEST)
  if(NOT RIME_FOUND)
    message(WARNING "Rime library not found. Test program cannot be built.")
    message(STATUS "To build test program, either:")
    message(STATUS "  1. Install Rime development package")
    message(STATUS "  2. Set RIME_ROOT_DIR to Rime installation directory")
    message(STATUS "  3. Ensure find_package(rime) can find Rime")
  else()
    # glog should already be found above, but check again for test program
    if(NOT glog_FOUND)
      message(WARNING "glog not found. Test program may not compile.")
    endif()
    
    # Create test executable
    add_executable(test_bert_grammar
      src/test_bert_grammar.cc
      $<TARGET_OBJECTS:rime-bert-grammar-objs>
    )
    
    # Set runtime library for test executable to match glog (static /MT)
    if(MSVC)
      if(POLICY CMP0091)
        set_target_properties(test_bert_grammar PROPERTIES
          MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
      else()
        # For older CMake, set flags directly on target
        set_target_properties(test_bert_grammar PROPERTIES
          COMPILE_FLAGS_RELEASE "/MT"
          COMPILE_FLAGS_DEBUG "/MTd")
      endif()
    endif()
    
    # Link libraries - Rime
    if(TARGET rime::rime)
      target_link_libraries(test_bert_grammar rime::rime)
    elseif(RIME_LIBRARIES)
      target_link_libraries(test_bert_grammar ${RIME_LIBRARIES})
    else()
      # Try to find Rime library manually
      if(WIN32)
        # On Windows, .lib files are usually in build/src/Release or build/Release
        find_library(RIME_LIB_FOR_TEST
          NAMES rime rime_static
          PATHS 
            "${RIME_ROOT_DIR}/build/src/Release"
            "${RIME_ROOT_DIR}/build/src/Debug"
            "${RIME_ROOT_DIR}/build/bin/Release"
            "${RIME_ROOT_DIR}/build/bin/Debug"
            "${RIME_ROOT_DIR}/build/Release"
            "${RIME_ROOT_DIR}/build/Debug"
            "${RIME_ROOT_DIR}/build/lib/Release"
            "${RIME_ROOT_DIR}/build/lib/Debug"
            "${RIME_ROOT_DIR}/build/lib"
            "${RIME_ROOT_DIR}/build"
            "${RIME_ROOT_DIR}/lib"
            "${RIME_ROOT_DIR}/lib/Release"
            "${RIME_ROOT_DIR}/lib/Debug"
          NO_DEFAULT_PATH
        )
        # Also try without NO_DEFAULT_PATH to search system paths
        if(NOT RIME_LIB_FOR_TEST)
          find_library(RIME_LIB_FOR_TEST NAMES rime rime_static)
        endif()
      else()
        find_library(RIME_LIB_FOR_TEST
          NAMES rime rime_static
          PATHS "${RIME_ROOT_DIR}/build" "${RIME_ROOT_DIR}/lib"
          NO_DEFAULT_PATH
        )
        if(NOT RIME_LIB_FOR_TEST)
          find_library(RIME_LIB_FOR_TEST NAMES rime rime_static)
        endif()
      endif()
      if(RIME_LIB_FOR_TEST)
        target_link_libraries(test_bert_grammar ${RIME_LIB_FOR_TEST})
        message(STATUS "Linking Rime library: ${RIME_LIB_FOR_TEST}")
      else()
        message(WARNING "Rime library not found for linking.")
        message(STATUS "  You may need to build librime first:")
        message(STATUS "    cd ${RIME_ROOT_DIR}")
        message(STATUS "    mkdir build && cd build")
        message(STATUS "    cmake .. && cmake --build . --config Release")
        message(WARNING "Test program will fail to link without Rime library.")
      endif()
    endif()
    
    # Link libraries - glog
    if(TARGET glog::glog)
      target_link_libraries(test_bert_grammar glog::glog)
    elseif(GLOG_LIBRARY)
      target_link_libraries(test_bert_grammar ${GLOG_LIBRARY})
      message(STATUS "Linking glog library: ${GLOG_LIBRARY}")
    else()
      # Try to find glog library manually
      if(WIN32)
        find_library(GLOG_LIB_FOR_TEST
          NAMES glog glogd glog_static
          PATHS 
            "${RIME_ROOT_DIR}/deps/glog/build/Release"
            "${RIME_ROOT_DIR}/deps/glog/build/Debug"
            "${RIME_ROOT_DIR}/build/src/Release"
            "${RIME_ROOT_DIR}/build/src/Debug"
            "${RIME_ROOT_DIR}/build/deps/glog/src/Release"
            "${RIME_ROOT_DIR}/build/deps/glog/src/Debug"
            "${RIME_ROOT_DIR}/build/bin/Release"
            "${RIME_ROOT_DIR}/build/bin/Debug"
            "${GLOG_ROOT}/lib"
            "${GLOG_ROOT}/lib/Release"
            "${GLOG_ROOT}/lib/Debug"
            "${GLOG_ROOT}/build/Release"
            "${GLOG_ROOT}/build/Debug"
            "${RIME_ROOT_DIR}/deps/glog/lib"
            "${RIME_ROOT_DIR}/deps/glog/build/lib"
            "${RIME_ROOT_DIR}/build/deps/glog/lib"
          NO_DEFAULT_PATH
        )
        # Also try without NO_DEFAULT_PATH
        if(NOT GLOG_LIB_FOR_TEST)
          find_library(GLOG_LIB_FOR_TEST NAMES glog glogd glog_static)
        endif()
      else()
        find_library(GLOG_LIB_FOR_TEST
          NAMES glog glog_static
          PATHS "${GLOG_ROOT}/lib" "${RIME_ROOT_DIR}/deps/glog/lib"
          NO_DEFAULT_PATH
        )
        if(NOT GLOG_LIB_FOR_TEST)
          find_library(GLOG_LIB_FOR_TEST NAMES glog glog_static)
        endif()
      endif()
      if(GLOG_LIB_FOR_TEST)
        target_link_libraries(test_bert_grammar ${GLOG_LIB_FOR_TEST})
        message(STATUS "Linking glog library: ${GLOG_LIB_FOR_TEST}")
      else()
        message(WARNING "glog library not found for linking.")
        message(STATUS "  glog may need to be built. Check: ${RIME_ROOT_DIR}/deps/glog")
        message(WARNING "Test program will fail to link without glog library.")
      endif()
    endif()
    
    if(GLOG_INCLUDE_DIR)
      target_include_directories(test_bert_grammar PRIVATE ${GLOG_INCLUDE_DIR})
    endif()
    
    # Add glog compile definitions for newer versions
    target_compile_definitions(test_bert_grammar PRIVATE
      GLOG_NO_ABBREVIATED_SEVERITIES)
    
    if(ONNXRUNTIME_FOUND)
      target_include_directories(test_bert_grammar PRIVATE
        ${ONNXRUNTIME_INCLUDE_DIRS})
      target_link_libraries(test_bert_grammar
        ${ONNXRUNTIME_LIBRARIES})
    endif()
    
    target_include_directories(test_bert_grammar PRIVATE
      ${RIME_INCLUDE_DIRS}
      ${CMAKE_CURRENT_SOURCE_DIR}/src)
    
    if(Boost_FOUND AND Boost_INCLUDE_DIRS)
      target_include_directories(test_bert_grammar PRIVATE
        ${Boost_INCLUDE_DIRS})
    endif()
    
    if(Boost_FOUND AND Boost_LIBRARIES)
      target_link_libraries(test_bert_grammar
        ${Boost_LIBRARIES})
    endif()
    
    # Link Windows system libraries needed by glog
    if(WIN32)
      target_link_libraries(test_bert_grammar
        dbghelp  # Required by glog for symbol resolution
      )
    endif()
    
    # Add compile definitions
    if(ONNXRUNTIME_FOUND)
      target_compile_definitions(test_bert_grammar PRIVATE RIME_USE_ONNXRUNTIME)
    endif()
    
    message(STATUS "Test program will be built: test_bert_grammar")
  endif()
endif()

