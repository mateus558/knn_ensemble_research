#include "Experiment/Experiment.hpp"
#include "Experiment/FileUtils.hpp"
#include "SimulatedAnnealing/SimulatedAnnealing.hpp"
#include "SimulatedAnnealing/kNNEnsembleW.hpp"

void Experiment::add_parameters() {
    this->add_option("-o,--output", this->results_folder, "Folder where the experiments results will be saved.")
        ->group("Experiment");
    this->add_option("-p,--data", this->data_folder, "Folder where the datasets are stored.")
        ->group("Experiment");
    this->add_option("-t,--threads", this->max_threads, "The maximum number of threads used for paralelization.")
        ->group("Experiment");
    this->add_option<std::vector<std::string>>("-d,--datasets", this->datasets)
        ->description("Datasets that will be used during the experiment, remember that they must be at the location defined by the parameter --data.")
        ->group("Experiment")
        ->delimiter(',');
    this->add_option("-f,--folds", this->n_folds, "Number of folds utilized to measure SA performace metrics.")
        ->group("Experiment");

    this->add_option<std::vector<size_t>>("-k,--ks", this->ks, "knn k values that will be used by the knn ensemble.")
        ->group("Simulated Annealing (SA)")
        ->delimiter(',');
    this->add_option("-n,--n_folds", this->sa_folds, "Number of folds utilized to obtain objective function value on SA.")
        ->group("Simulated Annealing (SA)");
    this->add_option("-T,--temperature", this->sa_temp, "Temperature that SA must start.")
        ->group("Simulated Annealing (SA)");
    this->add_option("-m,--min_temp_iter", this->min_temp_iter, "Minimum number of iterations that SA must execute in a temperature.")
        ->group("Simulated Annealing (SA)");
    this->add_option("-a,--alpha", this->alpha, "Decay of the temperature on SA, where, for example, 0.9 means that it will decay 10\% at each temperature change.")
        ->group("Simulated Annealing (SA)");
}

void Experiment::run() {  	
    this->threads = (this->n_folds > this->max_threads) ? this->max_threads : this->n_folds;
    mltk::Timer timer;
    std::error_code err;

    if(!createPath(this->results_folder, err)) {
        std::cout << "Failed to create path, err: " << err.message() << std::endl;
        return;
    }

    for(size_t i = 0; i < datasets.size(); i++) {
        mltk::Data<double> data((data_folder + datasets[i]).c_str());	
        std::map<std::string, std::shared_ptr<mltk::metrics::dist::BaseMatrix>> distances;

        for(const size_t k: this->ks){
            mltk::ensemble::kNNEnsembleW<double> ensemb(k);

            ensemb.setSamples(mltk::make_data<double>(data));
            
            if(distances.size() == 0){
                std::cout << "[" << data.name() << "] Pre computing distances for all metrics." << std::endl;
                distances = ensemb.computeDistances();
            }

            std::cout << "[" << data.name() << "] Creating " + std::to_string(n_folds) + "-fold partitions." << std::endl;
            auto folds = mltk::validation::kfoldsplit(data, n_folds);
            
            std::string dataset_results_folder = results_folder + data.name() + "/" + std::to_string(k) + "/";
                
            if(!createPath(dataset_results_folder, err)) {
                std::cout << "Failed to create path, err: " << err.message() << std::endl;
                continue;
            }

            auto partial_kfold = [folds, k, distances, dataset_results_folder, this](const int a, const int b) {
                std::vector<FoldResult> _errors(b - a);
                for(size_t i = a; i < b; i++){
                    auto fold = folds[i];

                    std::error_code err;
                    std::string data_results_folder = dataset_results_folder + "exec_" +  std::to_string(fold.execution) + "_fold_" + std::to_string(fold.fold) + "/";
                
                    if(!createPath(data_results_folder, err)) {
                        std::cout << "Failed to create path, err: " << err.message() << std::endl;
                        continue;
                    }
                    std::ofstream json(data_results_folder + "results_fold" + std::to_string(i+1) + ".json");

                    fold.train.write(data_results_folder + "train", "csv");
                    fold.test.write(data_results_folder + "test", "csv");

                    auto results = this->evaluate_fold(fold, k, distances, json);
                    
                    _errors[i - a] = results;
                    json.close();
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

    mltk::Point<std::future<std::vector<FoldResult>>> futuresPool(numBatches);
    for(size_t i = 0; i < numBatches; i++){
        size_t start = i * batchSize;
        size_t end = start + batchSize;
        
        futuresPool[i] = std::async(std::launch::async, partial_kfold, start, end);
    }

    mltk::Point<FoldResult> results(totalTasks);
    mltk::Point<size_t> errors(totalTasks);
    mltk::Timer timer1;

    for(size_t i = 0; i < futuresPool.size(); i++) {
        auto foldResult = futuresPool[i].get();
        
        for(size_t j = 0; j < foldResult.size(); j++) {
            results[i + j] = foldResult[j];
            errors[i + j] = foldResult[j].errors["sa"];
        }
    }

    size_t total_errors = errors.sum();
    size_t accuracy = 1.0 - total_errors/(double)data.size();

    auto elapsed = timer1.elapsed();

    std::cout << std::endl;
    std::cout << "[" << data.name() << "]" << "Total errors: " << total_errors << std::endl;
    std::cout << "[" << data.name() << "]" << "Accuracy: " << accuracy * 100 << std::endl;
    std::cout << "[" << data.name() << "]" << "Execution time: " << elapsed << " ms" << std::endl;
}

FoldResult Experiment::evaluate_fold(mltk::validation::TrainTestPair<double> fold, size_t k, 
            std::map<std::string, std::shared_ptr<mltk::metrics::dist::BaseMatrix>> distances, std::ofstream &results_json) {
    
    mltk::Timer timer;
    mltk::SimulatedAnnealing sa(fold.train, k, this->sa_folds, this->sa_temp, this->alpha, this->min_temp_iter);   
    FoldResult fold_results;

    for(auto& matrix_pair: distances) {
        sa.setDistanceMatrix(matrix_pair.first, matrix_pair.second);
    }

    auto results = sa.optimize();
    auto weights = results.first;
    auto report = results.second;

    fold_results.sa_duration = timer.elapsed();

    mltk::ensemble::kNNEnsembleW<double> knn_ensemb(k);
    
    knn_ensemb.setSamples(mltk::make_data<double>(fold.train));
    knn_ensemb.setWeights(weights.X());                
    for(auto& matrix_pair: distances) {
        knn_ensemb.setDistanceMatrix(matrix_pair.first, matrix_pair.second);
    }

    std::vector<std::string> metrics = knn_ensemb.metricsNames();
    
    for(auto& point: fold.test.points()) {
        double pred = knn_ensemb.evaluate(*point);
        
        if(pred != point->Y()) {
            fold_results.errors_ids["sa"].push_back(point->Id());
            fold_results.errors["sa"]++;
        }

        for(auto& metric: metrics) {
            double pred = knn_ensemb.evaluate(*point, metric);
            
            if(pred != point->Y()) {
                fold_results.errors_ids[metric].push_back(point->Id());
                fold_results.errors[metric]++;
            }
        }
    }

    fold_results.weights = weights;
    fold_results.fold = fold.fold;
    fold_results.execution = fold.execution;
    fold_results.k = k;

    double acc = 1.0 - (double)fold_results.errors["sa"]/fold.test.size();

    report["execution_time"] = fold_results.sa_duration;
    report["accuracy"] = acc;
    report["errors"] = fold_results.errors["sa"];
    report["errors_ids"] = fold_results.errors_ids["sa"];
    report["fold"] = fold.fold;

    results_json << report.dump(4) << std::endl;

    return fold_results;
}