# Set options
SET (YAML_CPP_BUILD_CONTRIB OFF CACHE BOOL "")
SET (YAML_CPP_BUILD_TOOLS OFF CACHE BOOL "")
SET (YAML_BUILD_SHARED_LIBS OFF CACHE BOOL "")

ADD_SUBDIRECTORY (yaml-cpp)

SET (TENSORSTREAM_EXTERNAL_LIBRARY_INCLUDES
    ${CMAKE_CURRENT_SOURCE_DIR}/yaml-cpp/include
    ${CMAKE_CURRENT_SOURCE_DIR}/eigen
PARENT_SCOPE)

SET (TENSORSTREAM_EXTERNAL_LIBRARY_LINKS
    yaml-cpp
PARENT_SCOPE)
