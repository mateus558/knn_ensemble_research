//
// Created by mateus on 03/04/2021.
//
#include <iostream>
#include <execution>
#include <tuple>
#include <fstream>
#include <ufjfmltk.hpp>
#include <map>
#include "kNNEnsemble.hpp"
#define DATASET_FOLDER "../datasets/"

template<typename Data, typename Model>
double estimate_ensemble_acc(Data& data, Model& ensemble){
    return mltk::validation::kkfold(data, ensemble, 10, 10, 42).accuracy;
}

int main(){
    std::vector<std::string> datasets = {"pima.data", "sonar.data", "bupa.data", "wdbc.data", "ionosphere.data", "biodegradetion.csv",
                                         "vehicle.csv", "ThoraricSurgery.arff", "seismic-bumps.arff"};
    bool at_end[10] = {false,false,false,false,false,false,false,true, true,false};
    std::vector<size_t> ks = {3, 5, 7};
    std::vector<std::pair<std::string, bool>> datasets_pair;
    std::vector<std::tuple<std::string, size_t, double>> results;

    for(int i = 0; i < datasets.size(); i++){
        datasets_pair.emplace_back(datasets[i], at_end[i]);
    }

    std::for_each(std::execution::par, datasets_pair.begin(), datasets_pair.end(), [&results, ks](auto& dataset_pair){
        mltk::Data<double> data("../datasets/" + dataset_pair.first, dataset_pair.second);

        std::clog << dataset_pair.first << std::endl;

        std::for_each(std::execution::par, ks.begin(), ks.end(), [&data, dataset_pair, &results](size_t k){
            mltk::ensemble::kNNEnsemble<double> knn(data, k);
            knn.set_test(true);
            double acc = estimate_ensemble_acc(data, knn);

            std::cout << dataset_pair.first << "-- k = " << k << "-- acc = "  << acc << std::endl;
            results.emplace_back(dataset_pair.first, k, acc);
        });
    });

    std::ofstream file("max_acc_log/max_accs_HV.csv");

    if(!file.is_open()){
        std::cerr << "Error opening results file!" << std::endl;
        return 1;
    }

    for(auto result: results){
        file << std::get<0>(result) << "," << std::get<1>(result) << "," << std::get<2>(result) << std::endl;
    }
    file.close();
}