set(ONNXRUNTIME_ROOT "" CACHE PATH "Path to an extracted ONNX Runtime C/C++ SDK")

if(NOT ONNXRUNTIME_ROOT AND DEFINED ENV{ONNXRUNTIME_ROOT})
  file(TO_CMAKE_PATH "$ENV{ONNXRUNTIME_ROOT}" ONNXRUNTIME_ROOT)
endif()

find_path(ONNXRUNTIME_INCLUDE_DIR
  NAMES onnxruntime_cxx_api.h
  HINTS "${ONNXRUNTIME_ROOT}/include"
  REQUIRED
)

find_library(ONNXRUNTIME_LIBRARY
  NAMES onnxruntime onnxruntime.lib libonnxruntime
  HINTS "${ONNXRUNTIME_ROOT}/lib"
  REQUIRED
)

add_executable(aeronode_runtime src/main.cpp)
target_include_directories(aeronode_runtime PRIVATE "${ONNXRUNTIME_INCLUDE_DIR}")
target_link_libraries(aeronode_runtime PRIVATE "${ONNXRUNTIME_LIBRARY}")

if(MSVC)
  target_compile_options(aeronode_runtime PRIVATE /W4 /permissive-)
else()
  target_compile_options(aeronode_runtime PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(WIN32)
  add_custom_command(TARGET aeronode_runtime POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      "${ONNXRUNTIME_ROOT}/lib/onnxruntime.dll"
      "$<TARGET_FILE_DIR:aeronode_runtime>"
    COMMENT "Copying ONNX Runtime DLL beside aeronode_runtime"
  )
endif()
