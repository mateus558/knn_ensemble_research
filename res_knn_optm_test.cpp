//
// Created by Marim on 10/10/2021.
//

#include <iostream>
#include <sstream>

#include "KNNEnsembleOptm.h"
#include "utils.h"

thread_pool pool1(10);
thread_pool kfold_pool(12);

std::pair<double, mltk::Point<double>> custom_kkfold(mltk::Data<double> &data, int k = 3, int ensemb_folds = 10, double valid_prop = 0.5, int qtde = 10,
                     int n_folds = 10, size_t seed = 0){
    double kkfold_acc = 0;
    mltk::Point<double> kkind_accs(7, 0.0);

    auto kfold = [&kkfold_acc, &kkind_accs, valid_prop, k, ensemb_folds](mltk::Data<double> data, int n_folds, size_t seed){
        auto folds = mltk::validation::kfoldsplit(data, n_folds, true, seed);
        double kfold_acc = 0;
        mltk::Point<double> kind_accs(7, 0.0);

        for(auto fold: folds){
            auto train_valid = mltk::validation::partTrainTest(fold.train, int(1.0/((valid_prop < 0.5)?valid_prop:1-valid_prop)), true, seed);
            mltk::ensemble::KNNEnsembleOptm ensemb((valid_prop > 0.5)?train_valid.train:train_valid.test, k, false, ensemb_folds);

            ensemb.train();
            std::cout << ensemb.getWeights() << std::endl;
            ensemb.set_samples((valid_prop > 0.5)?train_valid.test:train_valid.train);

            auto preds = ensemb.batchEvaluate(fold.test);
            double fold_acc = 0.0;
            for(int j = 0; j < preds.size(); j++){
                if(preds[j] == fold.test(j).Y()){
                    fold_acc += 1;
                }
            }
            ensemb.set_samples(fold.train);
            kind_accs += ensemb.accs_on_learners(fold.test);
            fold_acc /= fold.test.size();
            kfold_acc += fold_acc;
        }
        kind_accs /= n_folds;
        kfold_acc /= n_folds;
        mutex.lock();
            kkfold_acc += kfold_acc;
            kkind_accs += kind_accs;
        mutex.unlock();
    };
    for(int i = 0; i < qtde; i++){
        kfold_pool.push_task(kfold, data, n_folds, seed+i);
    }
    kfold_pool.wait_for_tasks();
    kkfold_acc /= qtde;
    kkind_accs /= qtde;

    return std::make_pair(kkfold_acc, kkind_accs);
}

int main(){
    std::vector<std::string> datasets = {
                                         "vehicle.csv"};
    bool at_end[] = {false};
//    std::vector<std::string> datasets = {"sonar.data", "vehicle.csv"};
//    bool at_end[] = {false, false};
    std::vector<mltk::LearnerPointer<double>> learners;
    std::vector<size_t> ks = {3, 5, 7};
    std::ofstream csv("../results/results_optim_new_method_80_201.csv");

    if(!csv.is_open()){
        std::cerr << "Error opening output file." << std::endl;
        return 1;
    }
    csv << "sep=;" << std::endl;
    csv << "Dataset;k;Accuracy;Individual accs" << std::endl;
    synced_stream csv_out(csv);
    synced_stream sync_out;

    int i = 0;
    for(const auto& data_path: datasets){
        auto data = load_dataset(data_path, "../datasets/", at_end[i++]);
//        auto train_test = mltk::validation::partTrainTest(data, 2);
//        train_test.train.setName(data.name());
        std::cout << std::endl;
        for(size_t k: ks){
            auto experiment = [&sync_out, &csv_out](mltk::Data<double> data, size_t k){
                auto acc = custom_kkfold(data, k, 10, 0.5);

                mutex.lock();
                sync_out.println("Dataset: ", data.name());
                sync_out.println("K value: ", k);
                sync_out.println("Accuracy: ", acc.first);
                sync_out.println("Indivicual accs: ", acc.second);
                sync_out.println("----------------------------------------");
                std::stringstream ss;
                ss << acc.second;
                std::string accs = ss.str();
                csv_out.println(data.name(),";"+std::to_string(k),";"+std::to_string(acc.first)
                                +";"+accs);
                mutex.unlock();
            };

//            auto experiment1 = [&sync_out, &csv_out](mltk::Data<double> data, size_t k){
//                mltk::ensemble::KNNEnsembleOptm ensemb(data, k, false, 3, false);
//                auto report = mltk::validation::kkfold(data, ensemb, 10, 10);
//
//                auto laccs = ensemb.getAccs();
//                mutex.lock();
//                sync_out.println("Dataset: ", data.name());
//                sync_out.println("Test size: ", data.size());
//                sync_out.println("Train size: ", data.size());
//                sync_out.println("K value: ", k);
//                sync_out.println("Weights: ", ensemb.getWeights());
//                sync_out.println("Accuracy: ", report.accuracy);
//                sync_out.println("Indivicual accs: ", laccs);
//                sync_out.println("----------------------------------------");
//                std::stringstream ss;
//                ss << ensemb.getWeights();
//                std::string w = ss.str();
//                ss.str("");
//                ss.clear();
//                ss << laccs;
//                std::string accs = ss.str();
//                csv_out.println(data.name(),";"+std::to_string(k),";"+std::to_string(report.accuracy)
//                                                                        +";" +w+";"+accs);
//                mutex.unlock();
//            };
            pool1.push_task(experiment, data, k);
            //pool1.push_task(experiment1, data, k);
        }
    }
    pool1.wait_for_tasks();
    return 0;
}