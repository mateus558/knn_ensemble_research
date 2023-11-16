#include <iostream>
#include <ufjfmltk/ufjfmltk.hpp>
#include "SimulatedAnnealing.hpp"
#include "kNNEnsembleW.hpp"
#include "thread_pool.hpp"
#include "utils.h"
#include "globals.h"
#include <filesystem>

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

void test() {
    thread_pool pool(2);
    mltk::Data samples = mltk::datasets::make_blobs(1500).dataset;
    auto data_split = mltk::validation::partTrainTest(samples, 3);
    std::vector<size_t> ks = {3, 5, 7};

    mltk::SimulatedAnnealing sa(data_split.train, 7);    

    auto results = sa.optimize();

    mltk::ensemble::kNNEnsembleW<double> knn_ensemb(7);
    knn_ensemb.setWeights(results.first.X());

    std::cout << "Accuracy: " << mltk::validation::kkfold(data_split.test, knn_ensemb, 10, 10).accuracy << std::endl;

    std::cin.get();
}

void experiment() {
    thread_pool exp_pool(8);
    std::vector<std::string> datasets = {"bupa.data", "pima.data", "sonar.data", "ionosphere.data", "biodegradation.csv", "wdbc.data", "ThoraricSurgery.arff", 
                                         "seismic-bumps.arff", "vehicle.csv"};
    //std::vector<std::string> datasets = {"bupa.data"};
    //                                      "biodegradation.csv"};
    bool at_end[] = {false, false, false, false, false, false, true, true, false};

    auto load = [&exp_pool](const std::string& dataset, bool at_end) {
        std::cout << "Loading dataset: " << dataset << std::endl;
        std::string results_path = "../results/" + removeExtension(dataset);
        createFolder(results_path);
        synced_cout.println("dataset: " + dataset);
        auto data = load_dataset(dataset, DATA_PATH, at_end);
        auto data_split = mltk::validation::partTrainTest(data, 3);
        std::vector<size_t> ks = {1, 3, 5, 7};
        //std::vector<size_t> ks = {7};

        for(auto k : ks){
            auto experiment = [results_path](mltk::validation::TrainTestPair<double> data_split, size_t k) {
                auto start = std::chrono::high_resolution_clock::now();
                std::ofstream json(results_path + "/results_k_" + std::to_string(k) + ".json");
                mltk::SimulatedAnnealing sa(data_split.train, k, 10, 8, 0.9, 50);    

                auto results = sa.optimize();
                auto json_report = results.second;

                mltk::ensemble::kNNEnsembleW<double> knn_ensemb(k);
                knn_ensemb.setWeights(results.first.X());

                // Stop the clock
                auto end = std::chrono::high_resolution_clock::now();

                // Calculate the duration in milliseconds
                std::chrono::duration<double, std::milli> duration = end - start;
                double executionTime = duration.count();
                
                json_report["execution_time"] = executionTime;

                double acc = mltk::validation::kkfold(data_split.test, knn_ensemb, 10, 10).accuracy;
                std::cout << "Accuracy: " << acc << std::endl;

                json_report["test_accuracy"] = acc;

                json << json_report.dump(4) << std::endl;
            };

            exp_pool.push_task(experiment, data_split, k);
        }
    };

    
    for(int i = 0; i < datasets.size(); i++){
        load(datasets[i], at_end[i]);
    }
}


int main() {
    experiment();
    return 0;
}
