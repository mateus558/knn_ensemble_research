#include <iostream>
#include <chrono>
#include <mutex>
#include <future>

#include "KNNEnsembleOptmDSM.h"
#include "utils.h"
#include "thread_pool.hpp"

std::ofstream output;
std::ofstream output_csv;

synced_stream sync_out;
synced_stream sync_log(output);
synced_stream sync_csv(output_csv);

void prepare_out_files(){
    output.open("../results/execution/output_qpoptm_only_one_opt_dsm.txt");
    output_csv.open("../results/execution/results_qpoptm_only_one_opt_dsm.csv");

    if(!output.is_open() || !output_csv.is_open()){
        std::cerr << "The output file could not be open." << std::endl;
        exit(1);
    }

    output_csv << "sep=;" << std::endl;
    //output_csv << "Dataset;k;alpha;accuracy;weights;accuracies;time" << std::endl;
    std::cout << "sep=;\nDataset;k;alpha;accuracy;weights;time" << std::endl;
}

int main(int argc, char* argv[]){
    auto experiment = [&](const std::string& dataset, bool at_end, int id){
        auto data = load_dataset(dataset, "../datasets/", at_end, false);
        std::vector<std::pair<mltk::Data<double>, size_t>> ks;

        ks.emplace_back(data.copy(), 3);
        ks.emplace_back(data.copy(), 5);
        ks.emplace_back(data.copy(), 7);

        std::for_each(ks.begin(), ks.end(), [&](std::pair<mltk::Data<double>, size_t>& data_pair){
            auto run_valid = [&](const std::pair<mltk::Data<double>, size_t>& data_pair){
                auto alphas = mltk::linspace(0, 1, 11);
                alphas.resize(alphas.size()+1);
                alphas[alphas.size()-1] = 1;

                auto alpha_run = [](double alpha, std::pair<mltk::Data<double>, size_t> data_pair){
                    mltk::ensemble::KNNEnsembleOptm<double> knn_ensemb(data_pair.first, data_pair.second, alpha,
                                                                       false, 10, 0, 0);
                    mltk::Timer timer;

                    auto report = mltk::validation::kkfold(data_pair.first, knn_ensemb, 10, 10, true,
                                                           0, 0);

//                    sync_out.println("\n------------------------------------------------------");
//                    sync_out.println(data_pair.first.name(), " report\n");
//                    sync_out.println("k value: ", data_pair.second);
//                    sync_out.println("alpha: ", alpha);
//                    sync_out.println("accuracy: ", report.accuracy);
//                    sync_out.println("ensemble accuracies: ", knn_ensemb.getAccs());
//                    sync_out.println("ensemble weights: ", knn_ensemb.getWeights());
//                    sync_out.println("MSE: ", knn_ensemb.getMse());
//                    sync_out.println("\nvalidation exec. time: ", timer.elapsed() * 0.001, " s");
//                    sync_out.println("------------------------------------------------------");
//                    sync_log.println("\n------------------------------------------------------");
//                    sync_log.println(data_pair.first.name(), " report\n");
//                    sync_log.println("k value: ", data_pair.second);
//                    sync_log.println("alpha: ", alpha);
//                    sync_log.println("accuracy: ", report.accuracy);
//                    sync_log.println("ensemble accuracies: ", knn_ensemb.getAccs());
//                    sync_log.println("ensemble weights: ", knn_ensemb.getWeights());
//                    sync_log.println("MSE: ", knn_ensemb.getMse());
//                    sync_log.println("\nvalidation exec. time: ", timer.elapsed() * 0.001, " s");
//                    sync_log.println("------------------------------------------------------");
                    sync_out.println(data_pair.first.name(), ";", data_pair.second, ";", alpha, ";", report.accuracy,
                                     ";", knn_ensemb.getWeights(),";",timer.elapsed() * 0.001);
                };
                for(auto alpha: alphas) {
                    pool.push_task(alpha_run, alpha, data_pair);
                }
            };
            pool.push_task(run_valid, data_pair);
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

    //std::cout << timer.elapsed()*0.001 << " s to compute." << std::endl;
    output.close();
    output_csv.close();
    std::cin.get();
    return EXIT_SUCCESS;
}