#pragma once

#include <ufjfmltk/ufjfmltk.hpp>

namespace mltk {
    class SimulatedAnnealing {
    private:
        double T = 1;
        double minT = 1E-4;
        double alpha = 0.9;
        double max_iterations = 100;
        size_t k = 7;
        std::vector<mltk::Point<double>> searchSpace;
        mltk::Data<double> samples;

        double metropolis(double diff, double t);

        std::pair<mltk::Point<double>, size_t> next(int neighbors, int curr_pos);

        double objective(mltk::Point<double> weights);

        std::vector<mltk::Point<double>> generateSearchSpace(size_t size, double max_weight, double step);

    public:
        SimulatedAnnealing() = default;

        explicit SimulatedAnnealing(const mltk::Data<double> &data, size_t k, double max_weight, double step, double T = 10, double alpha = 0.9, size_t max_iterations = 100);

        Point<double> optimize();
    };

};