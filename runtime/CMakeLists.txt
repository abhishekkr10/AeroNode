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

find_path(ZEROMQ_INCLUDE_DIR
  NAMES zmq.h zmq.hpp
  HINTS "C:/msys64/ucrt64/include" "$ENV{MINGW_PREFIX}/include"
  REQUIRED
)

find_library(ZEROMQ_LIBRARY
  NAMES zmq libzmq
  HINTS "C:/msys64/ucrt64/lib" "$ENV{MINGW_PREFIX}/lib"
  REQUIRED
)

find_path(NLOHMANN_JSON_INCLUDE_DIR
  NAMES nlohmann/json.hpp
  HINTS "C:/msys64/ucrt64/include" "$ENV{MINGW_PREFIX}/include"
  REQUIRED
)

add_executable(aeronode_runtime src/main.cpp)
target_include_directories(aeronode_runtime PRIVATE
  "${ONNXRUNTIME_INCLUDE_DIR}"
  "${CMAKE_CURRENT_SOURCE_DIR}/include"
  "${ZEROMQ_INCLUDE_DIR}"
  "${NLOHMANN_JSON_INCLUDE_DIR}"
)
target_link_libraries(aeronode_runtime PRIVATE
  "${ONNXRUNTIME_LIBRARY}"
  "${ZEROMQ_LIBRARY}"
  ws2_32
  iphlpapi
)

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

add_executable(test_entropy "${CMAKE_CURRENT_SOURCE_DIR}/../tests/test_entropy.cpp")
target_include_directories(test_entropy PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")
if(MSVC)
  target_compile_options(test_entropy PRIVATE /W4 /permissive-)
else()
  target_compile_options(test_entropy PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_executable(test_state_receiver "${CMAKE_CURRENT_SOURCE_DIR}/../tests/test_state_receiver.cpp")
target_include_directories(test_state_receiver PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/include"
  "${ZEROMQ_INCLUDE_DIR}"
  "${NLOHMANN_JSON_INCLUDE_DIR}"
)
target_link_libraries(test_state_receiver PRIVATE
  "${ZEROMQ_LIBRARY}"
  ws2_32
  iphlpapi
)
if(MSVC)
  target_compile_options(test_state_receiver PRIVATE /W4 /permissive-)
else()
  target_compile_options(test_state_receiver PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_executable(test_exit_decision "${CMAKE_CURRENT_SOURCE_DIR}/../tests/test_exit_decision.cpp")
target_include_directories(test_exit_decision PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/include"
  "${NLOHMANN_JSON_INCLUDE_DIR}"
)
if(MSVC)
  target_compile_options(test_exit_decision PRIVATE /W4 /permissive-)
else()
  target_compile_options(test_exit_decision PRIVATE -Wall -Wextra -Wpedantic)
endif()

