//
// Created by Marim on 21/02/2022.
//
#include "globals.h"
#include "utils.h"
#include "KNNEnsembleLin.hpp"

int main() {
    std::vector<std::string> datasets = {"pima.data", "sonar.data", "bupa.data", "wdbc.data", "ionosphere.data",
                                         "biodegradation.csv", "ThoraricSurgery.arff", "seismic-bumps.arff",
                                         "vehicle.csv"};
    bool at_end[] = {false, false, false, false, false, false, true, true, false};

    auto load = [](const std::string& dataset, bool at_end) {
        synced_cout.println("dataset: " + dataset);
        auto data = load_dataset(dataset, DATA_PATH, at_end);
        mltk::ensemble::KNNEnsembleLin knn(data, 5);

        knn.train();
    };

    for(int i = 0; i < datasets.size(); i++){
        pool.push_task(load, datasets[i], at_end[i]);
    }
}
