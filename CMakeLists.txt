cmake_minimum_required(VERSION 3.15)
project(graph-io LANGUAGES CXX)

option(GRAPH_IO_PARALLEL "Build parallel graph-io library" ON)
option(GRAPH_IO_GENERATORS "Build with graph generators" ON)
option(GRAPH_IO_MMAP "memory map binary files" ON)

if (GRAPH_IO_PARALLEL)
  find_package(MPI)
  if (NOT MPI_FOUND)
    message(WARNING "MPI not found, not building parallel components of library")
    set(GRAPH_IO_PARALLEL OFF)
  endif()
  if (GRAPH_IO_GENERATORS)
    if(NOT TARGET KaGen::KaGen)
      set(KAGEN_BUILD_APPS OFF CACHE BOOL "" FORCE)
      set(KAGEN_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
      set(KAGEN_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
      set(KAGEN_BUILD_TESTS OFF CACHE BOOL "" FORCE)
      include(FetchContent)
      FetchContent_Declare(
        KaGen
        GIT_REPOSITORY https://github.com/sebalamm/KaGen
        GIT_TAG        v1.3.0
        GIT_SUBMODULES_RECURSE TRUE
      )
      FetchContent_GetProperties(KaGen)
      if (NOT kagen_POPULATED)
        FetchContent_Populate(KaGen)
        add_subdirectory(${kagen_SOURCE_DIR} ${kagen_BINARY_DIR} EXCLUDE_FROM_ALL)
      endif()
    endif()
  endif()
endif()


#add_library(graph-io include/graph-io/distributed_graph_io.cpp include/graph-io/graph_io.cpp)
set (FILE_LIST
  "include/graph-io/graph_io.cpp"
  "include/graph-io/local_graph_view.cpp"
  )
if (GRAPH_IO_PARALLEL)
  list(APPEND FILE_LIST "include/graph-io/distributed_graph_io.cpp")
endif()

add_library(graph-io ${FILE_LIST})
target_include_directories(graph-io PUBLIC include)
if (GRAPH_IO_PARALLEL)
  target_compile_definitions(graph-io PUBLIC GRAPH_IO_PARALLEL)
  if (GRAPH_IO_GENERATORS)
    target_compile_definitions(graph-io PUBLIC GRAPH_IO_USE_KAGEN)
    target_link_libraries(graph-io PRIVATE KaGen::KaGen)
    endif()
  target_link_libraries(graph-io PUBLIC MPI::MPI_CXX)
endif()
if (GRAPH_IO_MMAP)
  target_compile_definitions(graph-io PRIVATE GRAPH_IO_MMAP)
endif()
target_compile_options(graph-io PRIVATE -std=c++17)

if (GRAPH_IO_GENERATORS)
    add_executable(generator_example generator_example.cpp)
    target_link_libraries(generator_example graph-io)
endif()
