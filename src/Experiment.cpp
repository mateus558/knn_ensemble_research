#include "Experiment/Experiment.hpp"
#include "Experiment/FileUtils.hpp"
#include "SimulatedAnnealing/SimulatedAnnealing.hpp"
#include "SimulatedAnnealing/kNNEnsembleW.hpp"


void Experiment::run() {
    std::cout << this->results_folder << std::endl;
    
}

size_t Experiment::evaluate_fold(mltk::validation::TrainTestPair<double> fold, size_t k, std::map<std::string, std::shared_ptr<mltk::metrics::dist::BaseMatrix>> distances) {
    mltk::Timer timer;
    mltk::SimulatedAnnealing sa(fold.train, k, 10, 9, 0.9, 50);   

    for(auto& matrix_pair: distances) {
        sa.setDistanceMatrix(matrix_pair.first, matrix_pair.second);
    }

    auto results = sa.optimize();
    auto weights = results.first;

    size_t sa_duration = timer.elapsed();

    mltk::ensemble::kNNEnsembleW<double> knn_ensemb(k);
    
    knn_ensemb.setSamples(mltk::make_data<double>(fold.train));
    knn_ensemb.setWeights(weights.X());                
    for(auto& matrix_pair: distances) {
        knn_ensemb.setDistanceMatrix(matrix_pair.first, matrix_pair.second);
    }

    size_t errors = 0;
    for(auto& point: fold.test.points()) {
        double pred = knn_ensemb.evaluate(*point);
        
        if(pred != point->Y()) {
            errors++;
        }
    }

    double acc = 1.0 - (double)errors/fold.test.size();

    return errors;
}

void Experiment::add_parameters() {
    this->add_option("-o,--output", this->results_folder, "Folder where the experiments results will be saved.");

}