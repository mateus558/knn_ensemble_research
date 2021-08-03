//
// Created by mateu on 30/07/2021.
//

#include <iostream>
#include "KNNEnsembleOptm.h"
#include "utils.h"

void experiment(const std::string& dataset, bool at_end, int k, const mltk::Point<double>& x0){
    auto data = load_dataset(dataset, "../datasets/", at_end);

    mltk::ensemble::KNNEnsembleOptm<double> knn_ensemb(data, k, true, 10, 42, 2);
    mltk::Timer timer;
    knn_ensemb.setStartingPoint(x0);
    knn_ensemb.train();
    std::cout << "ensemble accuracies: " << knn_ensemb.getAccs() << std::endl;
    std::cout << "ensemble weights: " << knn_ensemb.getWeights() << std::endl;
    auto report = mltk::validation::kkfold(data, knn_ensemb, 10, 10, true,
                                           42, 2);
    std::cout << data.name() << " report\n" << std::endl;
    std::cout << "k value: " << k << std::endl;
    std::cout << "accuracy: " << report.accuracy << std::endl;
    std::cout << "ensemble accuracies: " << knn_ensemb.getAccs() << std::endl;
    std::cout << "ensemble weights: " << knn_ensemb.getWeights() << std::endl;
    std::cout << "MSE: " << knn_ensemb.getMse() << std::endl;
    std::cout << "\nvalidation exec. time: " << timer.elapsed()*0.001 << " s" <<  std::endl;
    std::cout << "------------------------------------------------------\n";

}

int main(){
    //experiment("biodegradetion.csv", false, 3, {0,0,0,0.12,0.25,0.63});
    experiment("bupa.data", false, 3, {0.13, 0, 0.12, 0, 0.12, 0.25, 0.38});
    //experiment("ThoraricSurgery.arff", true, 3, {0,0,0.11,0,0,0,0.89});
    //experiment("ionosphere.data", false, 3, {0.38, 0, 0.12, 0, 0.12, 0, 0.38});
    //experiment("pima.data", false, 3, {0.26, 0, 0.26, 0.13, 0.13, 0, 0.26});

    std::cin.get();
}