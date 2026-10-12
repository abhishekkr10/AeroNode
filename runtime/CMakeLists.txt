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

add_library(aeronode_lib STATIC
  src/staged_model_runner.cpp
)
target_include_directories(aeronode_lib PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}/include"
  "${ONNXRUNTIME_INCLUDE_DIR}"
  "${ZEROMQ_INCLUDE_DIR}"
  "${NLOHMANN_JSON_INCLUDE_DIR}"
)
target_link_libraries(aeronode_lib PUBLIC
  "${ONNXRUNTIME_LIBRARY}"
)
if(MSVC)
  target_compile_options(aeronode_lib PRIVATE /W4 /permissive-)
else()
  target_compile_options(aeronode_lib PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_executable(aeronode_runtime src/main.cpp)
target_include_directories(aeronode_runtime PRIVATE
  "${ONNXRUNTIME_INCLUDE_DIR}"
  "${CMAKE_CURRENT_SOURCE_DIR}/include"
  "${ZEROMQ_INCLUDE_DIR}"
  "${NLOHMANN_JSON_INCLUDE_DIR}"
)
target_link_libraries(aeronode_runtime PRIVATE
  aeronode_lib
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

add_executable(test_staged_runner "${CMAKE_CURRENT_SOURCE_DIR}/../tests/test_staged_runner.cpp")
target_include_directories(test_staged_runner PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/include"
  "${ONNXRUNTIME_INCLUDE_DIR}"
)
target_link_libraries(test_staged_runner PRIVATE
  aeronode_lib
  "${ONNXRUNTIME_LIBRARY}"
)
if(MSVC)
  target_compile_options(test_staged_runner PRIVATE /W4 /permissive-)
else()
  target_compile_options(test_staged_runner PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(WIN32)
  add_custom_command(TARGET test_staged_runner POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      "${ONNXRUNTIME_ROOT}/lib/onnxruntime.dll"
      "$<TARGET_FILE_DIR:test_staged_runner>"
    COMMENT "Copying ONNX Runtime DLL beside test_staged_runner"
  )
endif()


add_executable(test_staged_execution "${CMAKE_CURRENT_SOURCE_DIR}/../tests/test_staged_execution.cpp")
target_link_libraries(test_staged_execution PRIVATE aeronode_lib)
if(MSVC)
  target_compile_options(test_staged_execution PRIVATE /W4 /permissive-)
else()
  target_compile_options(test_staged_execution PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(WIN32)
  add_custom_command(TARGET test_staged_execution POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      "${ONNXRUNTIME_ROOT}/lib/onnxruntime.dll"
      "$<TARGET_FILE_DIR:test_staged_execution>"
    COMMENT "Copying ONNX Runtime DLL beside test_staged_execution"
  )
endif()

# Existing suites use assert: keep assertions active even in Release builds.
add_executable(aeronode_benchmark "${PROJECT_SOURCE_DIR}/benchmarks/staged_benchmark.cpp")
target_link_libraries(aeronode_benchmark PRIVATE aeronode_lib)
if(MSVC)
  target_compile_options(aeronode_benchmark PRIVATE /W4 /permissive-)
else()
  target_compile_options(aeronode_benchmark PRIVATE -Wall -Wextra -Wpedantic)
endif()
if(WIN32)
  target_link_libraries(aeronode_benchmark PRIVATE psapi)
  add_custom_command(TARGET aeronode_benchmark POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${ONNXRUNTIME_ROOT}/lib/onnxruntime.dll"
      "$<TARGET_FILE_DIR:aeronode_benchmark>")
endif()

foreach(test_target test_entropy test_state_receiver test_exit_decision test_staged_runner test_staged_execution)
  if(MSVC)
    target_compile_options(${test_target} PRIVATE /UNDEBUG)
  else()
    target_compile_options(${test_target} PRIVATE -UNDEBUG)
  endif()
  add_test(NAME ${test_target} COMMAND ${test_target})
  set_tests_properties(${test_target} PROPERTIES WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}")
endforeach()

# Python supplies verification fixtures only; aeronode_lib has no Python dependency.
find_package(Python3 COMPONENTS Interpreter QUIET)
if(Python3_Interpreter_FOUND)
  add_test(NAME prepare_phase5d_fixtures
    COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/tests/prepare_phase5d_fixtures.py")
  set_tests_properties(prepare_phase5d_fixtures PROPERTIES
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}" FIXTURES_SETUP phase5d_data)
  set_tests_properties(test_staged_execution PROPERTIES FIXTURES_REQUIRED phase5d_data)
endif()
