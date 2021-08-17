//
// Created by Marim on 08/08/2021.
//

#include <ufjfmltk/ufjfmltk.hpp>
#include "kNNEnsembleWSS.hpp"
#include "utils.h"
#include <fstream>

std::ofstream output;
std::ofstream output_csv;

void prepare_out_files(){
    output.open("../results/execution/output_hv.txt");
    output_csv.open("../results/execution/results_hv.csv");

    if(!output.is_open() || !output_csv.is_open()){
        std::cerr << "The output file could not be open." << std::endl;
        exit(1);
    }

    output_csv << "sep=;" << std::endl;
    output_csv << "Dataset;k;accuracy;individual accuracies;time" << std::endl;
}

int main(){
    bool printed = false;
    auto experiment = [&printed](const std::string& dataset, bool at_end, int id){
        mutex.lock();
        std::cout << dataset << " " << at_end << std::endl;
        mutex.unlock();
        auto data = load_dataset(dataset, "../datasets/", at_end, false);
        std::string log_fname = data.name() + ".log", csv_fname = data.name() + ".csv";
        std::vector<std::pair<mltk::Data<double>, size_t>> ks;

        ks.emplace_back(data.copy(), 3);
        ks.emplace_back(data.copy(), 5);
        ks.emplace_back(data.copy(), 7);

        mutex.lock();
        if(!printed) {
            std::cout << "------------------------------------------------------------------" << std::endl;
            std::cout << " dataset  k  step    acc   weights   best_weights     best_acc    " << std::endl;
            std::cout << "------------------------------------------------------------------" << std::endl;
            printed = true;
        }

        log_files[log_fname].open(log_fname);
        log_files[csv_fname].open(csv_fname);
        if(!log_files[log_fname].is_open() || !log_files[csv_fname].is_open()){
            std::cerr << "Error opening log files" << std::endl;
        }
        log_files[log_fname] << "------------------------------------------------------------------" << std::endl;
        log_files[log_fname] << " dataset  k  step    acc   weights   best_weights     best_acc    " << std::endl;
        log_files[log_fname] << "------------------------------------------------------------------" << std::endl;
        log_files[csv_fname] << "sep=;" << std::endl;
        log_files[csv_fname] << "dataset;k;step;acc;weights;best_weights;best_acc" << std::endl;

        mutex.unlock();

        std::vector<std::future<void>> futures(ks.size());
        std::transform(ks.begin(), ks.end(), futures.begin(), [](std::pair<mltk::Data<double>, size_t>& data_pair){
            auto run_valid = [](std::pair<mltk::Data<double>, size_t> data_pair){
                mltk::ensemble::kNNEnsembleWSS<double> knn_ensemb(data_pair.first, data_pair.second);
                mltk::Timer timer;

                knn_ensemb.train();
                knn_ensemb.optimizeSubWeights(data_pair.first, 9, 104);

                auto report = mltk::validation::kkfold(data_pair.first, knn_ensemb, 10,10, true,
                                                      0, 0);
                auto accs = knn_ensemb.getWeights();

                mutex.lock();
                std::cout << "\n------------------------------------------------------\n";
                output << "\n------------------------------------------------------\n";
                std::cout << "dataset: " << data_pair.first.name() << "\n" << std::endl;
                output << "dataset: " << data_pair.first.name() << "\n" << std::endl;
                std::cout << "k value: " << data_pair.second << std::endl;
                output << "k value: " << data_pair.second << std::endl;
                std::cout << "accuracy: " << report.accuracy << std::endl;
                output << "accuracy: " << report.accuracy << std::endl;
                std::cout << "individual accs: " << accs << std::endl;
                output << "individual accs: " << accs << std::endl;
                std::cout << "\nvalidation exec. time: " << timer.elapsed()*0.001 << " s" <<  std::endl;
                output << "\nvalidation exec. time: " << timer.elapsed()*0.001 << " s" <<  std::endl;
                std::cout << "------------------------------------------------------\n";
                output << "------------------------------------------------------\n";
                output_csv << data_pair.first.name() << ";" << data_pair.second << ";" << report.accuracy << ";" <<
                accs << ";" << timer.elapsed()*0.001 << " s" << std::endl;
                mutex.unlock();
            };
            return std::async(std::launch::async, run_valid, data_pair);
        });

    };

    std::vector<std::string> datasets = {"pima.data", "sonar.data", "bupa.data", "wdbc.data", "ionosphere.data",
                                         "biodegradation.csv", "ThoraricSurgery.arff", "seismic-bumps.arff",
                                         "vehicle.csv"};
    bool at_end[] = {false, false, false, false, false, false, true, true, false};
    mltk::Timer timer;

    prepare_out_files();

    timer.reset();

    run(datasets, at_end, experiment);

    std::cout << timer.elapsed()*0.001 << " s to compute." << std::endl;
    output.close();
    output_csv.close();
    std::cin.get();
}