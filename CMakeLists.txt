cmake_minimum_required(VERSION 3.20)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_RELGPQUIRED ON)
add_compile_options(-Wall)
add_compile_options(-Wextra)
add_compile_options(-O3)

# include_directories(${PROJECT_SOURCE_DIR}/SomeGoodTemplate)
add_executable(tmp QType.cpp)