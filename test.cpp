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
        auto data_split = mltk::validation::partTrainTest(samples, 3);

        //pool.parallelize_loop(0, 9, [&data, &errors, weights, k] (const int a, const int b) -> void {
        mltk::ensemble::kNNEnsembleW<double> knn_ensemb(k);
        knn_ensemb.setWeights(weights.X());
        
        std::cout << "test size: " << data_split.test.size() << std::endl;
        std::cout << "class dist: " << mltk::Point<size_t>(data_split.test.classesDistribution()) << std::endl;
       // std::cout << data_split.test << std::endl;

        size_t errors_count = 0;
        for(auto point: data_split.test){
            auto pred = knn_ensemb.evaluate(*point);
          //  std::cout << pred << " " << point->Y() << std::endl;
            if(pred != point->Y()){
                errors_count++;
            }
        }
        std::cout << "errors: " << errors_count << std::endl;
        std::cout << "acc: " << 1.0 - (double)errors_count/data_split.test.size() << std::endl;

        double acc = mltk::validation::kkfold(data_split.test, knn_ensemb, 10,10).accuracy/100;
        std::cout << "acc" << 0 << ": " << acc << std::endl;
        //errors[0] += 1.0 - acc;
        
        //}, 10); 

        return errors.sum()/errors.size();
    }


int main(){
    std::vector<std::string> datasets = {"bupa.data"};

    std::vector<std::vector<double>> weights = {{0.14285714285714285, 0.14285714285714285, 0.14285714285714285, 0.14285714285714285, 0.14285714285714285, 0.14285714285714285, 0.14285714285714285}};
    
    bool at_end[] = {false};
    size_t i = 0;
    
    for(std::string dataset: datasets){
        auto data = load_dataset(dataset, DATA_PATH,  at_end[i]);

        std::cout << "Accuracy: " << (1-kkfold(data, weights[i], 1))*100 << std::endl;

        i++;
    }
    

    return 0;
}

