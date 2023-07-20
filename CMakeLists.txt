cmake_minimum_required(VERSION 3.15)
project(simulated_annealing)
set(CMAKE_CXX_STANDARD 17)

include(FetchContent)

FetchContent_Declare(json URL https://github.com/nlohmann/json/releases/download/v3.11.2/json.tar.xz)
FetchContent_MakeAvailable(json)

include(FetchContent)
FetchContent_Declare(
        ufjfmltk
        # Specify the commit you depend on and update it regularly.
        GIT_REPOSITORY https://github.com/mateus558/UFJF-Machine-Learning-Toolkit.git
        GIT_TAG b16211e9a202ef3dd32ab55ebdde877f4fe6c553
)
FetchContent_MakeAvailable(ufjfmltk)



add_executable(simulated_annealing main.cpp SimulatedAnnealing1.cpp utils.cpp)
target_link_libraries(simulated_annealing ufjfmltk nlohmann_json::nlohmann_json)

add_executable(test test.cpp SimulatedAnnealing1.cpp utils.cpp)
target_link_libraries(test ufjfmltk nlohmann_json::nlohmann_json)