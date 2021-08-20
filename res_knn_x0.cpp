//
// Created by mateu on 30/07/2021.
//

#include <iostream>
#include <mutex>
#include "KNNEnsembleOptm.h"
#include "utils.h"
#include "thread_pool.hpp"

synced_stream sync_out;

auto experiment = [](const std::string& dataset, bool at_end, int k, const mltk::Point<double>& x0){
    auto data = load_dataset(dataset, "../datasets/", at_end);

    mltk::ensemble::KNNEnsembleOptm<double> knn_ensemb(data, k, true, 10, 42, 0);
    mltk::Timer timer;
    knn_ensemb.setStartingPoint(x0);
    knn_ensemb.train();

    auto report = mltk::validation::kkfold(data, knn_ensemb, 10, 10, true,
                                           0, 0);

    sync_out.println(data.name(), " report\n");
    sync_out.println("k value: ", k);
    sync_out.println("accuracy: ", report.accuracy);
    sync_out.println("ensemble accuracies: ", knn_ensemb.getAccs());
    sync_out.println("ensemble weights: ", knn_ensemb.getWeights());
    sync_out.println("MSE: ", knn_ensemb.getMse());
    sync_out.println("\nvalidation exec. time: ", timer.elapsed()*0.001, " s");
    sync_out.println("------------------------------------------------------");
};

int main(){
    thread_pool pool(10);
    std::string dataset[] = {"biodegradation.csv", "bupa.data", "ThoraricSurgery.arff", "ionosphere.data", "pima.data"};
    bool atEnd[] = {false, false, true, false, false};
    int k = 3;
    std::vector<mltk::Point<double>> x0 = {{0, 0, 0, 0.24, 0, 0, 0.76}, {0.12, 0.12, 0.12, 0, 0.25, 0, 0.39},
                                           {0,0,0,0,0,0,1}, {0, 0, 0, 0, 0.24, 0.38, 0.38},
                                           {0.25, 0.12, 0.38, 0, 0, 0, 0.25}};
    for(int i = 0; i < 5; i++){
        pool.push_task(experiment, dataset[i], atEnd[i], k, x0[i]);
    }
    std::cin.get();
}