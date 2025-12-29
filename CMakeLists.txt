project(rime-bert-grammar)
cmake_minimum_required(VERSION 3.10)

aux_source_directory(src bert_grammar_src)

# Option to enable/disable ONNX Runtime
option(ENABLE_ONNXRUNTIME "Enable ONNX Runtime support" ON)

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

add_library(rime-bert-grammar-objs OBJECT ${bert_grammar_src})
if(BUILD_SHARED_LIBS)
  set_target_properties(rime-bert-grammar-objs
    PROPERTIES
    POSITION_INDEPENDENT_CODE ON)
endif()

# Add include directories
if(ONNXRUNTIME_FOUND)
  target_include_directories(rime-bert-grammar-objs PRIVATE
    ${ONNXRUNTIME_INCLUDE_DIRS})
endif()

# Link libraries
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

