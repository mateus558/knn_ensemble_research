cmake_minimum_required(VERSION 3.19)
project(knn_research)

option(ENABLE_THREADS "Enable multithreaded algorithms" OFF)
set(BUILD_SHARED_LIBS OFF)

if(ENABLE_THREADS)
    message("Multithreaded algorithms enabled.")
    add_compile_definitions(THREADS_ENABLED)
endif(ENABLE_THREADS)

#set(BUILD_SHARED_LIBS OFF)

include(FetchContent)
FetchContent_Declare(
        ufjfmltk
        # Specify the commit you depend on and update it regularly.
        URL https://github.com/mateus558/UFJF-Machine-Learning-Toolkit/archive/refs/tags/v0.51.8-beta.tar.gz
)
FetchContent_MakeAvailable(ufjfmltk)

add_subdirectory(alglib)
add_subdirectory(armadillo-10.6.1)

set(CMAKE_CXX_STANDARD 17)

add_executable(knn_res_est_acc_wpdsm res_knn_est_acc_WPDSM.cpp)
add_executable(knn_optm res_knn_optm.cpp utils.h)
add_executable(knn_res_est_acc_hv res_knn_est_acc_HV.cpp)
add_executable(linalg_test linalg_test.cpp)
add_executable(threads threads.cpp)

find_package( Threads )

target_link_libraries(knn_res_est_acc_wpdsm ufjfmltk -ltbb ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries(knn_res_est_acc_hv ufjfmltk ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries(threads -ltbb ${CMAKE_THREAD_LIBS_INIT})

if(ENABLE_THREADS)
    target_link_libraries(knn_optm -std=c++17 ufjfmltk armadillo alglib -fopenmp ${CMAKE_THREAD_LIBS_INIT})
else(ENABLE_THREADS)
    target_link_libraries(knn_optm ufjfmltk armadillo alglib)
endif(ENABLE_THREADS)
target_link_libraries(linalg_test alglib ufjfmltk)
