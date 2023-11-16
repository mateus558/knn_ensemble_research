#include "Experiment/Experiment.hpp"
#include "Experiment/FileUtils.hpp"
#include "SimulatedAnnealing/SimulatedAnnealing.hpp"
#include "SimulatedAnnealing/kNNEnsembleW.hpp"


void Experiment::run() {  	
    this->threads = (this->n_folds > this->max_threads) ? this->max_threads : this->n_folds;
    std::vector<std::string> datasets = {"wine.data"};
    std::vector<size_t> ks = {1, 3, 5, 7};
    mltk::Timer timer;
    std::error_code err;

    if(!createPath(results_folder, err)) {
        std::cout << "Failed to create path, err: " << err.message() << std::endl;
        return;
    }

    for(size_t i = 0; i < datasets.size(); i++) {
        mltk::Data<double> data((data_folder + datasets[i]).c_str());	
        std::map<std::string, std::shared_ptr<mltk::metrics::dist::BaseMatrix>> distances;

        for(const size_t k: ks){
            mltk::ensemble::kNNEnsembleW<double> ensemb(k);

            ensemb.setSamples(mltk::make_data<double>(data));
            
            if(distances.size() == 0){
                std::cout << "[" << data.name() << "] Pre computing distances for all metrics." << std::endl;
                distances = ensemb.computeDistances();
            }

            std::cout << "[" << data.name() << "] Creating " + std::to_string(n_folds) + "-fold partitions." << std::endl;
            auto folds = mltk::validation::kfoldsplit(data, n_folds);
            
            std::string dataset_results_folder = results_folder + data.name() + "/";
                
            if(!createPath(dataset_results_folder, err)) {
                std::cout << "Failed to create path, err: " << err.message() << std::endl;
                continue;
            }

            auto partial_kfold = [folds, k, distances, dataset_results_folder, this](const int a, const int b) {
                mltk::Point<size_t> _errors(b - a);
                for(size_t i = a; i < b; i++){
                    auto fold = folds[i];

                    std::error_code err;
                    std::string data_results_folder = dataset_results_folder + "exec_" +  std::to_string(fold.execution) + "_fold_" + std::to_string(fold.fold) + "/";
                
                    if(!createPath(data_results_folder, err)) {
                        std::cout << "Failed to create path, err: " << err.message() << std::endl;
                        continue;
                    }

                    fold.train.write(data_results_folder + "train", "csv");
                    fold.test.write(data_results_folder + "test", "csv");

                    size_t errors = this->evaluate_fold(fold, k, distances);
                    _errors[i - a] = errors;
                }
                return _errors;
            };

            this->parallel_kfold(data, folds.size(), partial_kfold);                        
        }
    }
    auto elapsed = timer.elapsed();
    std::cout << "\n\nTotal execution time: " << elapsed << " ms" << std::endl;
}

template< typename Fn >
void Experiment::parallel_kfold(mltk::Data<double> &data, size_t totalTasks, Fn partial_kfold) {
    size_t numBatches = (totalTasks > this->threads) ? this->threads : totalTasks;
    size_t batchSize = std::ceil(totalTasks/double(numBatches));

    mltk::Point<std::future<mltk::Point<size_t>>> futuresPool(numBatches);
    for(size_t i = 0; i < numBatches; i++){
        size_t start = i * batchSize;
        size_t end = start + batchSize;
        
        futuresPool[i] = std::async(std::launch::async, partial_kfold, start, end);
    }

    mltk::Point<size_t> consolidatedResults(totalTasks);
    
    mltk::Timer timer1;
    for(size_t i = 0; i < futuresPool.size(); i++) {
        auto foldResult = futuresPool[i].get();
        
        for(size_t j = 0; j < foldResult.size(); j++) {
            consolidatedResults[i + j] = foldResult[j];
        }
    }

    size_t total_errors = consolidatedResults.sum();
    size_t accuracy = 1.0 - (double)total_errors/consolidatedResults.size();

    auto elapsed = timer1.elapsed();

    std::cout << std::endl;
    std::cout << "[" << data.name() << "]" << "Total errors: " << std::to_string(total_errors) << std::endl;
    std::cout << "[" << data.name() << "]" << "Accuracy: " << std::to_string(accuracy * 100) << std::endl;
    std::cout << "[" << data.name() << "]" << "Execution time: " << std::to_string(elapsed) << " ms" << std::endl;
}

size_t Experiment::evaluate_fold(mltk::validation::TrainTestPair<double> fold, size_t k, std::map<std::string, std::shared_ptr<mltk::metrics::dist::BaseMatrix>> distances) {
    mltk::Timer timer;
    mltk::SimulatedAnnealing sa(fold.train, k, this->sa_folds, this->sa_temp, this->alpha, this->min_temp_iter);   

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
    this->add_option("-d,--data", this->data_folder, "Folder where the datasets are stored.");
    this->add_option("-t,--threads", this->max_threads, "The maximum number of threads used for paralelization.");

    this->add_option("-f,--folds", this->n_folds, "Number of folds utilized to measure SA performace metrics.");

    this->add_option("--sa_folds", this->sa_folds, "Number of folds utilized to obtain objective function value on SA.");
    this->add_option("--sa_temp", this->sa_temp, "Initial temperature that SA must start.");
    this->add_option("--min_temp_iter", this->min_temp_iter, "Minimum number of iterations that SA must execute in a temperature.");
    this->add_option("--alpha", this->alpha, "Decay of the temperature on SA, where, for example, 0.9 means that it will decay 10\% at each temperature change.");
}