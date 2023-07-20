#include <iostream>
#include <ufjfmltk/ufjfmltk.hpp>
#include "SimulatedAnnealing1.hpp"
#include "kNNEnsembleW.hpp"
#include "thread_pool.hpp"
#include "utils.h"
#include "globals.h"

double kkfold(mltk::Data<double> samples, mltk::Point<double> weights, size_t k){
        thread_pool pool;
        mltk::Point<double> errors(10);
        std::vector<mltk::Data<double>> data(10);

        for(size_t i = 0; i < 10; i++){
            data[i] = samples.copy();
        }

        pool.parallelize_loop(0, 9, [&data, &errors, weights, k] (const int a, const int b) -> void {
            for(size_t i = a; i < b; i++){
                mltk::ensemble::kNNEnsembleW<double> knn_ensemb(data[i], k);
                knn_ensemb.setWeights(weights.X());
                
                double acc = mltk::validation::kfold(data[i], knn_ensemb, 10).accuracy/100;
                std::cout << "acc" << i << ": " << acc << std::endl;
                errors[i] += 1.0 - acc;
            }
        }, 10); 

        return errors.sum()/errors.size();
    }


int main(){
    std::vector<std::string> datasets = {"ThoraricSurgery.arff", "seismic-bumps.arff",
                                         "vehicle.csv"};

    std::vector<std::vector<double>> weights = {{0.14285714285714285, 0.14285714285714285, 0.14285714285714285, 0.14285714285714285, 0.14285714285714285, 0.14285714285714285, 0.14285714285714285}, 
                                                {0.14285714285714285, 0.14285714285714285, 0.14285714285714285, 0.14285714285714285, 0.14285714285714285, 0.14285714285714285, 0.14285714285714285}, 
                                                {0.14285714285714285, 0.14285714285714285, 0.14285714285714285, 0.14285714285714285, 0.14285714285714285, 0.14285714285714285, 0.14285714285714285}};
    
    bool at_end[] = {true, true, false};

    while(true){
        size_t i = 0;
        for(std::string dataset: datasets){
            auto data = load_dataset(dataset, DATA_PATH,  at_end[i]);

            std::cout << "Accuracy: " << (1-kkfold(data, weights[i], 3))*100 << std::endl;

            i++;
        }
    }

    return 0;
}

