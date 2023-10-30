cmake_minimum_required(VERSION 3.15)
project(simulated_annealing)


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(TBB REQUIRED COMPONENTS tbb)

include(FetchContent)

FetchContent_Declare(json URL https://github.com/nlohmann/json/releases/download/v3.11.2/json.tar.xz)
FetchContent_MakeAvailable(json)

FetchContent_Declare(
        ufjfmltk
        # Specify the commit you depend on and update it regularly.
        URL https://github.com/mateus558/UFJF-Machine-Learning-Toolkit/archive/refs/tags/V0.54.3-beta.tar.gz
)
FetchContent_MakeAvailable(ufjfmltk)

add_executable(simulated_annealing main.cpp SimulatedAnnealing1.cpp utils.cpp)
target_link_libraries(simulated_annealing tbb ufjfmltk nlohmann_json::nlohmann_json)

add_executable(ind_acc ind_accs.cpp SimulatedAnnealing1.cpp utils.cpp)
target_link_libraries(ind_acc tbb ufjfmltk nlohmann_json::nlohmann_json)

add_executable(test test.cpp SimulatedAnnealing1.cpp utils.cpp)
target_link_libraries(test tbb ufjfmltk nlohmann_json::nlohmann_json)