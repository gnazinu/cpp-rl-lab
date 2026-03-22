cmake_minimum_required(VERSION 3.20)

project(cpp_rl_lab VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(cpp_rl_lab
    src/main.cpp
)

target_include_directories(cpp_rl_lab
    PRIVATE
        include
)
