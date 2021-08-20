//
// Created by Marim on 10/08/2021.
//
#include <ufjfmltk/ufjfmltk.hpp>
#include "thread_pool.hpp"
#include "utils.h"

int main(){
    std::vector<std::string> datasets = {"pima.data", "sonar.data", "bupa.data", "wdbc.data", "ionosphere.data",
                                         "biodegradation.csv", "ThoraricSurgery.arff", "seismic-bumps.arff",
                                         "vehicle.csv"};
    bool at_end[] = {false, false, false, false, false, false, true, true, false};
    std::vector<mltk::LearnerPointer<double>> learners;
    std::vector<size_t> ks = {3, 5, 7};
    thread_pool pool(10);
    std::ofstream csv("../results/individual_accs.csv");

    if(!csv.is_open()){
        std::cerr << "Error opening output file." << std::endl;
        return 1;
    }
    csv << "sep=;" << std::endl;
    csv << "Dataset;Distance metric;k;Accuracy" << std::endl;
    synced_stream csv_out(csv);
    synced_stream sync_out;

    int i = 0;
    for(const auto& data_path: datasets){
        auto data = load_dataset(data_path, "../datasets/", at_end[i++]);
        std::cout << std::endl;
        for(size_t k: ks){
            auto euclidean = [&sync_out, &csv_out](mltk::Data<double> data, size_t k){
                mltk::classifier::KNNClassifier<double, mltk::metrics::dist::Euclidean<double>> knn(data, k);

                auto report = mltk::validation::kkfold(data, knn, 10, 10);

                sync_out.println("Dataset: ", data.name());
                sync_out.println("Distance metric: euclidean");
                sync_out.println("K value: ", k);
                sync_out.println("Accuracy: ", report.accuracy);
                sync_out.println("----------------------------------------");
                csv_out.println(data.name(), ";euclidean",";"+std::to_string(k),";"+std::to_string(report.accuracy));
            };
            auto lorentzian = [&sync_out, &csv_out](mltk::Data<double> data, size_t k){
                mltk::classifier::KNNClassifier<double, mltk::metrics::dist::Lorentzian<double>> knn(data, k);

                auto report = mltk::validation::kkfold(data, knn, 10, 10);

                sync_out.println("Dataset: ", data.name());
                sync_out.println("Distance metric: lorentzian");
                sync_out.println("K value: ", k);
                sync_out.println("Accuracy: ", report.accuracy);
                sync_out.println("----------------------------------------");
                csv_out.println(data.name(), ";lorentzian",";"+std::to_string(k),";"+std::to_string(report.accuracy));
            };
            auto cosine = [&sync_out, &csv_out](mltk::Data<double> data, size_t k){
                mltk::classifier::KNNClassifier<double, mltk::metrics::dist::Cosine<double>> knn(data, k);

                auto report = mltk::validation::kkfold(data, knn, 10, 10);

                sync_out.println("Dataset: ", data.name());
                sync_out.println("Distance metric: cosine");
                sync_out.println("K value: ", k);
                sync_out.println("Accuracy: ", report.accuracy);
                sync_out.println("----------------------------------------");
                csv_out.println(data.name(), ";cosine",";"+std::to_string(k),";"+std::to_string(report.accuracy));
            };
            auto bhattacharyya = [&sync_out, &csv_out](mltk::Data<double> data, size_t k){
                mltk::classifier::KNNClassifier<double, mltk::metrics::dist::Bhattacharyya<double>> knn(data, k);

                auto report = mltk::validation::kkfold(data, knn, 10, 10);

                sync_out.println("Dataset: ", data.name());
                sync_out.println("Distance metric: bhattacharyya");
                sync_out.println("K value: ", k);
                sync_out.println("Accuracy: ", report.accuracy);
                sync_out.println("----------------------------------------");
                csv_out.println(data.name(), ";bhattacharyya",";"+std::to_string(k),";"+std::to_string(report.accuracy));
            };
            auto pearson = [&sync_out, &csv_out](mltk::Data<double> data, size_t k){
                mltk::classifier::KNNClassifier<double, mltk::metrics::dist::Pearson<double>> knn(data, k);

                auto report = mltk::validation::kkfold(data, knn, 10, 10);

                sync_out.println("Dataset: ", data.name());
                sync_out.println("Distance metric: pearson");
                sync_out.println("K value: ", k);
                sync_out.println("Accuracy: ", report.accuracy);
                sync_out.println("----------------------------------------");
                csv_out.println(data.name(), ";pearson",";"+std::to_string(k),";"+std::to_string(report.accuracy));
            };
            auto kullbackLeibler = [&sync_out, &csv_out](mltk::Data<double> data, size_t k){
                mltk::classifier::KNNClassifier<double, mltk::metrics::dist::KullbackLeibler<double>> knn(data, k);

                auto report = mltk::validation::kkfold(data, knn, 10, 10);

                sync_out.println("Dataset: ", data.name());
                sync_out.println("Distance metric: kullbackLeibler");
                sync_out.println("K value: ", k);
                sync_out.println("Accuracy: ", report.accuracy);
                sync_out.println("----------------------------------------");
                csv_out.println(data.name(), ";kullbackLeibler",";"+std::to_string(k),";"+std::to_string(report.accuracy));
            };
            auto hassanat = [&sync_out, &csv_out](mltk::Data<double> data, size_t k){
                mltk::classifier::KNNClassifier<double, mltk::metrics::dist::Hassanat<double>> knn(data, k);

                auto report = mltk::validation::kkfold(data, knn, 10, 10);

                sync_out.println("Dataset: ", data.name());
                sync_out.println("Distance metric: hassanat");
                sync_out.println("K value: ", k);
                sync_out.println("Accuracy: ", report.accuracy);
                sync_out.println("----------------------------------------");
                csv_out.println(data.name(), ";hassanat",";"+std::to_string(k),";"+std::to_string(report.accuracy));
            };

            pool.push_task(euclidean, data, k);
            pool.push_task(lorentzian, data, k);
            pool.push_task(cosine, data, k);
            pool.push_task(bhattacharyya, data, k);
            pool.push_task(pearson, data, k);
            pool.push_task(kullbackLeibler, data, k);
            pool.push_task(hassanat, data, k);
        }
    }
    pool.wait_for_tasks();
}

