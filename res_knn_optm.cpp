//
// Created by mateus on 11/05/2021.
//

#include <iostream>
#include <chrono>
#include <mutex>
#include <future>

#include "KNNEnsembleOptm.h"
#include "utils.h"

std::ofstream output;
std::ofstream output_csv;

void prepare_out_files(){
    output.open("../results/execution/output_qpoptm_only_one_opt1.txt");
    output_csv.open("../results/execution/results_qpoptm_only_one_opt1.csv");

    if(!output.is_open() || !output_csv.is_open()){
        std::cerr << "The output file could not be open." << std::endl;
        exit(1);
    }

    output_csv << "sep=;" << std::endl;
    output_csv << "Dataset;k;accuracy;weights;time" << std::endl;
}

int main(int argc, char* argv[]){
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
                mltk::ensemble::KNNEnsembleOptm<double> knn_ensemb(data_pair.first, data_pair.second, false, 10, 0, 0);
                mltk::Timer timer;
                knn_ensemb.train();

                auto weights = knn_ensemb.getWeights();
                auto ind_accs = knn_ensemb.getAccs();
                auto mse = knn_ensemb.getMse();
                auto report = mltk::validation::kkfold(data_pair.first, knn_ensemb, 10, 10, true,
                                                       0, 0);

                mutex.lock();
                std::cout << "\n------------------------------------------------------\n";
                std::cout << data_pair.first.name() << " report\n" << std::endl;
                std::cout << "k value: " << data_pair.second << std::endl;
                std::cout << "accuracy: " << report.accuracy << std::endl;
                std::cout << "ensemble accuracies: " << ind_accs << std::endl;
                std::cout << "ensemble weights: " << weights << std::endl;
                std::cout << "MSE: " << mse << std::endl;
                std::cout << "\nvalidation exec. time: " << timer.elapsed()*0.001 << " s" <<  std::endl;
                std::cout << "------------------------------------------------------\n";
                output << "\n------------------------------------------------------\n";
                output << data_pair.first.name() << " report\n" << std::endl;
                output << "k value: " << data_pair.second << std::endl;
                output << "accuracy: " << report.accuracy << std::endl;
                output << "ensemble accuracies: " << ind_accs << std::endl;
                output << "ensemble weights: " << weights << std::endl;
                output << "MSE: " << knn_ensemb.getMse() << std::endl;
                output << "\nvalidation exec. time: " << timer.elapsed()*0.001 << " s" <<  std::endl;
                output << "------------------------------------------------------\n";
                output_csv <<  data_pair.first.name() << ";" << data_pair.second << ";" << report.accuracy <<
                ";" << weights << std::endl;
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
    return EXIT_SUCCESS;
}