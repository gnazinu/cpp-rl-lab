cmake_minimum_required(VERSION 3.20)

project(cpp_rl_lab VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_library(cpp_rl_lab_lib
    src/agents/q_learning_agent.cpp
    src/agents/random_agent.cpp
    src/cli/command_line.cpp
    src/env/maze_environment.cpp
    src/env/maze_layout.cpp
    src/metrics/metrics_logger.cpp
    src/training/trainer.cpp
    src/utils/filesystem.cpp
)

target_include_directories(cpp_rl_lab_lib
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)

target_compile_features(cpp_rl_lab_lib PUBLIC cxx_std_17)

if(MSVC)
    target_compile_options(cpp_rl_lab_lib PRIVATE /W4 /permissive-)
else()
    target_compile_options(cpp_rl_lab_lib PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_executable(cpp_rl_lab src/main.cpp)
target_link_libraries(cpp_rl_lab PRIVATE cpp_rl_lab_lib)

enable_testing()

add_executable(cpp_rl_lab_tests
    tests/test_main.cpp
    tests/test_agents.cpp
    tests/test_environment.cpp
    tests/test_maze_layout.cpp
    tests/test_training.cpp
)

target_link_libraries(cpp_rl_lab_tests PRIVATE cpp_rl_lab_lib)

if(MSVC)
    target_compile_options(cpp_rl_lab_tests PRIVATE /W4 /permissive-)
else()
    target_compile_options(cpp_rl_lab_tests PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_test(NAME cpp_rl_lab_tests COMMAND cpp_rl_lab_tests)
