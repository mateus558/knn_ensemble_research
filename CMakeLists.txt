cmake_minimum_required(VERSION 3.18)
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
        #URL https://github.com/mateus558/UFJF-Machine-Learning-Toolkit/archive/refs/tags/v0.51.9-beta.tar.gz
        GIT_REPOSITORY https://github.com/mateus558/UFJF-Machine-Learning-Toolkit.git
        GIT_TAG b16211e9a202ef3dd32ab55ebdde877f4fe6c553
)
FetchContent_MakeAvailable(ufjfmltk)

add_subdirectory(alglib-3.17.0)
add_subdirectory(armadillo-10.6.1)

set(CMAKE_CXX_STANDARD 17)

add_executable(knn_res_est_acc_wpdsm res_knn_est_acc_WPDSM.cpp)
add_executable(knn_optm res_knn_optm.cpp utils.h utils.cpp)
add_executable(knn_optm_test res_knn_optm_test.cpp utils.h utils.cpp)
add_executable(knn_hv res_knn_hv.cpp utils.cpp utils.h)
add_executable(knn_soft res_knn_softv.cpp utils.cpp)
add_executable(knn_lin res_knn_lin.cpp utils.cpp)
add_executable(knn_res_est_acc_hv res_knn_est_acc_HV.cpp)
add_executable(linalg_test linalg_test.cpp)
add_executable(threads threads.cpp utils.cpp)
add_executable(start_point_test res_knn_x0.cpp utils.cpp)
add_executable(knn_wss res_knn_wss.cpp utils.cpp)
add_executable(knn_ind res_knn_ind_acc.cpp utils.cpp)
add_executable(knn_optm_dsm res_knn_optm_dsm.cpp utils.cpp)


find_package( Threads )

target_link_libraries(knn_res_est_acc_wpdsm -pthread ufjfmltk ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries(knn_res_est_acc_hv -pthread ufjfmltk ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries(threads -pthread ${CMAKE_THREAD_LIBS_INIT} ufjfmltk)
target_link_libraries(knn_hv -pthread ufjfmltk armadillo)
target_link_libraries(knn_soft -pthread ufjfmltk)
target_link_libraries(knn_wss ufjfmltk -pthread ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries(knn_ind ufjfmltk -pthread ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries(knn_lin ufjfmltk armadillo alglib -pthread ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries(knn_optm ufjfmltk armadillo alglib -pthread ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries(knn_optm_test ufjfmltk armadillo alglib -pthread ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries(knn_optm_dsm ufjfmltk armadillo alglib -pthread ${CMAKE_THREAD_LIBS_INIT})


if(ENABLE_THREADS)
    target_link_libraries(start_point_test ufjfmltk alglib armadillo ${CMAKE_THREAD_LIBS_INIT})
else(ENABLE_THREADS)
    target_link_libraries(start_point_test ufjfmltk alglib armadillo)
endif(ENABLE_THREADS)
target_link_libraries(linalg_test alglib ufjfmltk)
