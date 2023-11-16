cmake_minimum_required(VERSION 3.15)
project(simulated_annealing)


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(TBB REQUIRED COMPONENTS tbb)

include(FetchContent)

FetchContent_Declare(json URL https://github.com/nlohmann/json/releases/download/v3.11.2/json.tar.xz)
FetchContent_MakeAvailable(json)

FetchContent_Declare(CLI11 URL https://github.com/CLIUtils/CLI11/archive/refs/tags/v2.3.2.tar.gz)
FetchContent_MakeAvailable(CLI11)

FetchContent_Declare(
        ufjfmltk
        # Specify the commit you depend on and update it regularly.
        URL https://github.com/mateus558/UFJF-Machine-Learning-Toolkit/archive/refs/tags/V0.56.2-beta.tar.gz
)
FetchContent_MakeAvailable(ufjfmltk)

add_executable(experiment experiment.cpp SimulatedAnnealing.cpp utils.cpp)
target_link_libraries(experiment tbb CLI11 ufjfmltk nlohmann_json::nlohmann_json)

add_executable(simulated_annealing main.cpp SimulatedAnnealing.cpp utils.cpp)
target_link_libraries(simulated_annealing tbb ufjfmltk nlohmann_json::nlohmann_json)

add_executable(ind_acc ind_accs.cpp SimulatedAnnealing.cpp utils.cpp)
target_link_libraries(ind_acc tbb ufjfmltk nlohmann_json::nlohmann_json)

add_executable(test test.cpp SimulatedAnnealing.cpp utils.cpp)
target_link_libraries(test tbb ufjfmltk nlohmann_json::nlohmann_json)