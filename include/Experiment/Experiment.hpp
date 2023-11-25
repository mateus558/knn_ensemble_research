#pragma once

#include <ufjfmltk/ufjfmltk.hpp>
#include <CLI/CLI.hpp>


class FoldResult {
    public:
        std::map<std::string, size_t> errors;
        std::map<std::string, std::vector<size_t>> errors_ids;
        std::map<std::string, double> accs;
        mltk::Point<double> weights;
        size_t fold;
        size_t execution;
        size_t k;
        size_t sa_duration;
};

class Experiment: public CLI::App {
    private:
        std::string results_folder{"../results/"};
        std::string data_folder{"../datasets/"};
        size_t max_threads{std::thread::hardware_concurrency()};
        size_t threads{std::thread::hardware_concurrency()};
        
        std::vector<size_t> ks = {1};
        std::vector<std::string> datasets = {"iris.data"};
        size_t n_folds{10};
        
        size_t sa_folds{10};
        size_t sa_temp{9};
        size_t min_temp_iter{50};
        double alpha{0.9};

    public:
        using CLI::App::App;

        void add_parameters();

        void run();

    private:
        FoldResult evaluate_fold(mltk::validation::TrainTestPair<double> fold, size_t k, std::map<std::string, std::shared_ptr<mltk::metrics::dist::BaseMatrix>> distances, std::ofstream &results_json);

        template< typename Fn >
        void parallel_kfold(mltk::Data<double> &data, size_t totalTasks, Fn partial_kfold, const std::string &kfold_results_folder); 
};