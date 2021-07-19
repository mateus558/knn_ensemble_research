//
// Created by mateus on 11/05/2021.
//

#include <iostream>
#include <chrono>
#include "KNNEnsembleOptm.h"

using namespace std::chrono;

int main(int argc, char* argv[]){
    auto data = mltk::datasets::make_blobs(500, 2, 2).dataset;
    //mltk::Data<double> data("../datasets/blobs_3d.csv");
    mltk::Timer timer;

    mltk::visualize::Visualization<double> vis(data);
//    auto labels = data.getFeature(data.dim()-1);
//    data.removeFeatures({int(data.dim())});
//    data.setType("Regression");
//    for(int i = 0; i < data.size(); i++){
//        data.updatePointValue(i, labels[i]);
//    }
    mltk::ensemble::KNNEnsembleOptm<double> knn_ensem(data, 3);
    std::cout << "Dataset name: " << data.name() << std::endl;
    std::cout << "size: " << data.size() << std::endl;
    std::cout << "dims: " << data.dim() << std::endl;
    std::cout << "classes: " << mltk::Point<int>(data.classes()) << std::endl;
    std::cout << "classes distribution: " << mltk::Point<size_t>(data.classesDistribution()) << std::endl;
    std::cout << std::endl;
    for(int i = 0; i < 5; i++){
        std::cout << data(i) <<std::endl;
    }
    timer.reset();
    knn_ensem.train();
    std::cout << timer.elapsed() << "ms to compute." << std::endl;
    vis.plot2D();
    std::cin.get();
    timer.reset();
    knn_ensem.setVerbose(0);
    auto report = mltk::validation::kkfold(data, knn_ensem, 10, 10);
    std::cout << "accuracy = " << report.accuracy << std::endl;
    std::cout << timer.elapsed() << "ms to compute." << std::endl;
    return EXIT_SUCCESS;
}