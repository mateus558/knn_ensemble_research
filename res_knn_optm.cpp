//
// Created by mateus on 11/05/2021.
//

#include <iostream>
#include <chrono>
#include <mutex>
#include <future>

#include "KNNEnsembleOptm.h"
#include "utils.h"

auto experiment = [](const std::string& dataset, bool at_end, int id){
    auto data = load_dataset(dataset, "../datasets/", at_end);
    std::vector<std::pair<mltk::Data<double>, size_t>> ks;
    std::vector<std::future<void>> futures(3);

    ks.emplace_back(data.copy(), 3);
    ks.emplace_back(data.copy(), 5);
    ks.emplace_back(data.copy(), 7);

    std::transform(ks.begin(), ks.end(), futures.begin(), [](std::pair<mltk::Data<double>, size_t>& data_pair){
        auto run_valid = [](std::pair<mltk::Data<double>, size_t> data_pair){
            mltk::ensemble::KNNEnsembleOptm<double> knn_ensemb(data_pair.first, data_pair.second, 10, 0, 0);
            mltk::Timer timer;

            auto report = mltk::validation::kkfold(data_pair.first, knn_ensemb, 10, 10, true,
                                                   0, 0);

            mutex.lock();
            std::cout << "\n------------------------------------------------------\n";
            std::cout << data_pair.first.name() << " report\n" << std::endl;
            std::cout << "k value: " << data_pair.second << std::endl;
            std::cout << "accuracy: " << report.accuracy << std::endl;
            std::cout << "ensemble accuracies: " << knn_ensemb.getAccs() << std::endl;
            std::cout << "ensemble weights: " << knn_ensemb.getWeights() << std::endl;
            std::cout << "\nvalidation exec. time: " << timer.elapsed()/100 << " s" <<  std::endl;
            std::cout << "------------------------------------------------------\n";
            mutex.unlock();
        };
        return std::async(std::launch::async, run_valid, data_pair);
    });
};

int main(int argc, char* argv[]){
    std::vector<std::string> datasets = {"pima.data", "sonar.data", "bupa.data", "wdbc.data", "ionosphere.data",
                                         "biodegradetion.csv", "vehicle.csv", "ThoraricSurgery.arff"};
    bool at_end[] = {false,false,false,false,false,false,false,true, true,false};
    mltk::Timer timer;

    run(datasets, at_end, experiment);

    std::cout << timer.elapsed() << "ms to compute." << std::endl;
    std::cin.get();
    return EXIT_SUCCESS;
}