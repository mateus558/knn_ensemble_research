#include <iostream>
#include <ufjfmltk/ufjfmltk.hpp>
#include "SimulatedAnnealing1.hpp"
#include "kNNEnsembleW.hpp"
#include "thread_pool.hpp"
#include "utils.h"
#include "globals.h"
#include <filesystem>
#include <unordered_map>
#include <fstream>

#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace fs = std::filesystem;

std::string removeExtension(const std::string& filename) {
    // Find the last dot in the filename
    size_t lastDotPos = filename.find_last_of(".");
    
    // Check if a dot was found and it is not the first character
    if (lastDotPos != std::string::npos && lastDotPos != 0) {
        return filename.substr(0, lastDotPos);
    }
    
    // If no dot or dot is the first character, return the original filename
    return filename;
}

bool createFolder(const std::string& folderName) {
    if (fs::exists(folderName)) {
        std::cout << "Folder already exists." << std::endl;
        return false;
    }

    if (fs::create_directory(folderName)) {
        std::cout << "Folder created successfully." << std::endl;
        return true;
    } else {
        std::cout << "Failed to create folder." << std::endl;
        return false;
    }
}

std::vector<fs::path> collectFilePaths(const fs::path& directoryPath) {
    std::vector<fs::path> filePaths;

    for (const auto& entry : fs::directory_iterator(directoryPath)) {
        if (entry.is_directory()) {
            // Recur    sively collect file paths from subdirectories
            auto subdirectoryPaths = collectFilePaths(entry.path());
            filePaths.insert(filePaths.end(), subdirectoryPaths.begin(), subdirectoryPaths.end());
        } else if (entry.is_regular_file()) {
            // Store the file path
            filePaths.push_back(entry.path());
        }
    }

    return filePaths;
}

std::vector<std::pair<fs::path, json>> readAndParseFiles(const std::vector<fs::path>& filePaths) {
    std::vector<std::pair<fs::path, json>> jsonDataMap(filePaths.size());
    
    std::transform(std::execution::par_unseq, filePaths.begin(), filePaths.end(), jsonDataMap.begin(), [](const fs::path& filePath) {
        // Read and parse the JSON file        
        std::ifstream file(filePath);
        if (file.is_open()) {
            json jsonDataFromFile;
            try {
                file >> jsonDataFromFile;
            } catch (json::parse_error& e) {
                std::cerr << "Failed to parse JSON file " << filePath << ": " << e.what() << std::endl;
            }
            file.close();
            // Add the parsed JSON data to the map with the file path as the key
            return std::make_pair(filePath, jsonDataFromFile);
        } else {
            std::cerr << "Failed to open JSON file: " << filePath << std::endl;
        }
        return std::make_pair(filePath, json());
    });

    return jsonDataMap;
}

std::vector<std::pair<fs::path, json>> readResults(const fs::path& root){
    if (fs::exists(root) && fs::is_directory(root)) {
        std::vector<fs::path> files = collectFilePaths(root);
        return readAndParseFiles(files);
    } else {
        std::cerr << "Invalid directory path. path = " << root << std::endl;
    }
    return {};
}

mltk::Point<double> getWeights(const json& jsonData) {
    mltk::Point<double> weights(jsonData["weights"].size());
    std::transform(jsonData["weights"].begin(), jsonData["weights"].end(), weights.begin(), [](const auto& w) {
        return w;
    });
    return weights;
}

std::pair<std::string, bool> findDataset(const std::string& datasetName) {
    std::vector<std::string> datasets = {"bupa.data", "pima.data", "sonar.data", "ionosphere.data", "biodegradation.csv", "wdbc.data", "ThoraricSurgery.arff", 
                                         "seismic-bumps.arff", "vehicle.csv"};
    //std::vector<std::string> datasets = {"bupa.data"};
    //                                      "biodegradation.csv"};
    bool at_end[] = {false, false, false, false, false, false, true, true, false};
    
    for (size_t i = 0; i < datasets.size(); ++i) {
        if (removeExtension(datasets[i]) == datasetName) {
            return std::make_pair(datasets[i], at_end[i]);
        }
    }

    return std::make_pair("", false);
}


int main() {
    
    auto results = readResults("../results/");

    std::cout << "JSON data found:" << std::endl;
    for (auto& pair : results) {
        const fs::path& filePath = pair.first;
        json jsonData = pair.second;

        std::cout << std::endl;

        // Process each JSON data object along with its file path as needed
        std::cout << "File: " << filePath << std::endl;
         // Get the filename without the extension
        fs::path fileNameWithoutExtension = filePath.stem();

        // Go one directory up to get the dataset name
        fs::path datasetName = filePath.parent_path().filename();
        
        std::cout << "Dataset: " << datasetName << std::endl;

        int k = jsonData["k"];
        mltk::Point<double> weights = getWeights(jsonData);

        std::cout << "K: " << k << std::endl;
        std::cout << "Weights: " << weights << std::endl;

        mltk::ensemble::kNNEnsembleW<double> knn_ensemb(k);
        
        auto dataset = findDataset(datasetName);

        std::cout << "Dataset: " << dataset.first << std::endl;
        std::cout << "At end: " << dataset.second << std::endl;
        std::cout << "DATA_PATH: " << DATA_PATH << std::endl;

        auto data = load_dataset(dataset.first, "../datasets/", dataset.second);
        auto data_split = mltk::validation::partTrainTest(data, 3);

        knn_ensemb.setSamples(mltk::make_data<double>(data_split.test));
        mltk::Point ind_accs = knn_ensemb.individualAccuracies();

        jsonData["individual_accuracies"] = ind_accs.X();

        std::cout << "Individual Accuracies: " << ind_accs << std::endl;
        std::cout << "Ensemble test acc: " << jsonData["test_accuracy"] << std::endl;

        std::ofstream file(filePath);

        file << jsonData.dump(4) << std::endl;

        file.close();
        std::cout << std::endl;
    }

    return 0;
}
