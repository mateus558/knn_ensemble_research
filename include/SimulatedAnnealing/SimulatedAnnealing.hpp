#pragma once

#include <ufjfmltk/ufjfmltk.hpp>
#include <nlohmann/json.hpp>

#include "SimulatedAnnealing/kNNEnsembleW.hpp"

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
        size_t folds {10};
        std::vector<mltk::Point<double>> searchSpace;
        mltk::Data<double> samples;
        std::vector<std::unique_ptr<mltk::ensemble::kNNEnsembleW<double>>> ensembles;
        std::map<std::string, std::shared_ptr<metrics::dist::BaseMatrix>> distanceMatrices;

        mltk::Point<double> neighbour(mltk::Point<double> w);
        double objective(std::vector<mltk::Data<double>> &data, mltk::Point<double> weights);

    public:
        SimulatedAnnealing() = default;

        explicit SimulatedAnnealing(const mltk::Data<double> &data, size_t k, size_t folds = 10, double T = 2, double alpha = 0.9, size_t minTempIter = 10);

        std::pair<mltk::Point<double>, json> optimize();

        void setDistanceMatrix(std::string metric, std::shared_ptr<mltk::metrics::dist::BaseMatrix> distanceMatrix) {
            this->distanceMatrices[metric] = distanceMatrix;
        }
    };

};