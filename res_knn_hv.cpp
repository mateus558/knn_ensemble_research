//
// Created by Marim on 06/08/2021.
//

#include <ufjfmltk/ufjfmltk.hpp>
#include "kNNEnsemble.hpp"
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
    auto experiment = [](const std::string& dataset, bool at_end, int id){
        mutex.lock();
        std::cout << dataset << " " << at_end << std::endl;
        mutex.unlock();
        auto data = load_dataset(dataset, "../datasets/", at_end);
        std::vector<std::pair<mltk::Data<double>, size_t>> ks;

        ks.emplace_back(data.copy(), 3);
        ks.emplace_back(data.copy(), 5);
        ks.emplace_back(data.copy(), 7);

        std::vector<std::future<void>> futures(ks.size());
        std::transform(ks.begin(), ks.end(), futures.begin(), [](std::pair<mltk::Data<double>, size_t>& data_pair){
            auto run_valid = [](std::pair<mltk::Data<double>, size_t> data_pair){
                mltk::ensemble::kNNEnsemble<double> knn_ensemb(data_pair.first, data_pair.second);

                mltk::Timer timer;

                auto report = mltk::validation::kfold(data_pair.first, knn_ensemb, 10, true,
                                                       0, 0);
                auto accs = knn_ensemb.individual_accs();

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