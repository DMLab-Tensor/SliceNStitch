CMAKE_MINIMUM_REQUIRED (VERSION 3.10)

SET (CMAKE_CXX_STANDARD 17)

PROJECT (SliceNStitch)

# Project settings
SET (TENSORSTREAM_BUILD_TARGET "SliceNStitch")

ADD_SUBDIRECTORY (${CMAKE_CURRENT_SOURCE_DIR}/lib)

FILE (GLOB_RECURSE TENSORSTREAM_CPP_FILES ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)

# Build
ADD_EXECUTABLE (${TENSORSTREAM_BUILD_TARGET}
    ${TENSORSTREAM_CPP_FILES}
)

TARGET_INCLUDE_DIRECTORIES (${TENSORSTREAM_BUILD_TARGET} SYSTEM
PRIVATE
    ${TENSORSTREAM_EXTERNAL_LIBRARY_INCLUDES}
)

TARGET_LINK_LIBRARIES (${TENSORSTREAM_BUILD_TARGET}
    ${TENSORSTREAM_EXTERNAL_LIBRARY_LINKS}
)

# Compile options
TARGET_COMPILE_OPTIONS (${TENSORSTREAM_BUILD_TARGET} PRIVATE
    $<$<CONFIG:RELEASE>: -march=native>
)
