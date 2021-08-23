//
// Created by mateus on 19/07/2021.
//

#include <iostream>
#include <algorithm>
#include <utility>
#include "utils.h"
#include "kNNEnsembleW.hpp"

synced_stream sync_out;

int main(){
    auto valid = [](mltk::Data<double> data, size_t k, std::vector<double> weights){
        mltk::ensemble::kNNEnsembleW<double> knn_ensemb(data, k);

        knn_ensemb.setWeights(std::move(weights));

        auto report = mltk::validation::kkfold(data, knn_ensemb, 10, 10);


        sync_out.println("k value: ", k);
        sync_out.println("accuracy: ", report.accuracy);
        sync_out.println();
    };

    auto data = load_dataset("../datasets/biodegradation.csv");

    std::vector<std::vector<double>> weights = {{{0, 0, 0, 26, 0, 0, 78}}, {13, 0, 0, 13, 13, 13, 52},
                                                {26, 0, 0, 13, 0, 13, 52}};

    pool.push_task(valid, data.copy(), 3, weights[0]);
    pool.push_task(valid, data.copy(), 5, weights[1]);
    pool.push_task(valid, data.copy(), 7, weights[2]);

    std::cin.get();
    pool.wait_for_tasks();
}
