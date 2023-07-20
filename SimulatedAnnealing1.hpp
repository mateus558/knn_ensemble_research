#pragma once

#include <ufjfmltk/ufjfmltk.hpp>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace mltk {
    class SimulatedAnnealing {
    private:
        double T = 1;
        double minT = 1E-4;
        const double epslon = 1E-4;
        double alpha = 0.9;
        size_t k = 7;
        size_t minTempIter = 10;
        std::vector<mltk::Point<double>> searchSpace;
        mltk::Data<double> samples;

        mltk::Point<double> neighbour(mltk::Point<double> w);

        double objective(mltk::Point<double> weights);

    public:
        SimulatedAnnealing() = default;

        explicit SimulatedAnnealing(const mltk::Data<double> &data, size_t k, double T = 2, double alpha = 0.9, size_t minTempIter = 10);

        std::pair<mltk::Point<double>, json> optimize();
    };

};