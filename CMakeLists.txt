cmake_minimum_required(VERSION 3.24)

project (cuball CXX CUDA)

set(CMAKE_CXX_STANDARD 20)

find_package(glfw3 REQUIRED)
find_package(glm REQUIRED)
find_package(glad REQUIRED)

add_executable(cuball
                "src/cuball.cu"
)
target_link_libraries(cuball PRIVATE glfw glm::glm glad::glad)
