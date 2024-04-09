cmake_minimum_required(VERSION 3.20.0)
project(mrng VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(hip REQUIRED)
find_package(miopen)

function(target_link target_name)
  target_link_libraries(${target_name} PRIVATE xpmem)
  target_link_libraries(${target_name} PRIVATE hip::device)
  target_link_libraries(${target_name} PRIVATE hip::host MIOpen)
endfunction()

set(SOURCE_HEADERS ${CMAKE_CURRENT_SOURCE_DIR})

function(target_include target_name)
    target_include_directories(${target_name} PRIVATE ${SOURCE_HEADERS}/include)
endfunction()

add_subdirectory(examples)
