#pragma once

#include <ufjfmltk/ufjfmltk.hpp>
#include <CLI/CLI.hpp>


class Experiment: public CLI::App {
    private:
        std::string results_folder{"../results"};

    public:
        using CLI::App::App;

        void add_parameters();

        void run();

    private:
        size_t evaluate_fold(mltk::validation::TrainTestPair<double> fold, size_t k, std::map<std::string, std::shared_ptr<mltk::metrics::dist::BaseMatrix>> distances);
};