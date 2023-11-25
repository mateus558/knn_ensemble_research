//
// Created by mateuscmarim on 05/02/2021.
//

#ifndef UFJF_MLTK_KNNENSEMBLEW_HPP
#define UFJF_MLTK_KNNENSEMBLEW_HPP

#include <ufjfmltk/ufjfmltk.hpp>
#include <execution>
#include "thread_pool.hpp"


namespace mltk {
    namespace ensemble {
        template<typename T>
        class kNNEnsembleW: public Ensemble<T>, public classifier::Classifier<T> {
        private:
            size_t k;
            std::string voting_type = "soft";
            mltk::Point<double> weights;
            thread_pool pool{5};
            std::vector<std::string> metrics;
            std::map<std::string, std::shared_ptr<metrics::dist::BaseMatrix>> distanceMatrices;

        public:
            kNNEnsembleW() = default;

            kNNEnsembleW(size_t _k): k(_k) {
                this->m_learners.push_back(std::make_shared<classifier::KNNClassifier<T, metrics::dist::Euclidean<T>>>(k));
                metrics.push_back("Euclidean");
                
                this->m_learners.push_back(std::make_shared<classifier::KNNClassifier<T, metrics::dist::Lorentzian<T>>>(k));
                metrics.push_back("Lorentzian");
                
                this->m_learners.push_back(std::make_shared<classifier::KNNClassifier<T, metrics::dist::Cosine<T>>>(k));
                metrics.push_back("Cosine");
                
                this->m_learners.push_back(std::make_shared<classifier::KNNClassifier<T, metrics::dist::Bhattacharyya<T>>>(k));
                metrics.push_back("Bhattacharyya");
                
                this->m_learners.push_back(std::make_shared<classifier::KNNClassifier<T, metrics::dist::Pearson<T>>>(k));
                metrics.push_back("Pearson");
                
                this->m_learners.push_back(std::make_shared<classifier::KNNClassifier<T, metrics::dist::KullbackLeibler<T>>>(k));
                metrics.push_back("KullbackLeibler");
                
                this->m_learners.push_back(std::make_shared<classifier::KNNClassifier<T, metrics::dist::Hassanat<T>>>(k));
                metrics.push_back("Hassanat");
            }

            bool train() override{

                return true;
            }

            void setDistanceMatrix(std::string metric, std::shared_ptr<metrics::dist::BaseMatrix> distanceMatrix) {
                auto it = std::find(this->metrics.begin(), this->metrics.end(), metric);

                if (it != this->metrics.end()) {
                    auto idx = it - this->metrics.begin();
                    auto learner = this->m_learners[idx];
                    std::string metric = this->metrics[idx];

                    if(metric == "Euclidean") {
                        auto knn = this->castToKNN<metrics::dist::Euclidean>(learner);
                        knn->setPrecomputedDistances(*distanceMatrix);
                    } else if(metric == "Lorentzian") {
                        auto knn = this->castToKNN<metrics::dist::Lorentzian>(learner);
                        knn->setPrecomputedDistances(*distanceMatrix);
                    } else if(metric == "Cosine") {
                        auto knn = this->castToKNN<metrics::dist::Cosine>(learner);
                        knn->setPrecomputedDistances(*distanceMatrix);
                    } else if(metric == "Bhattacharyya") {
                        auto knn = this->castToKNN<metrics::dist::Bhattacharyya>(learner);
                        knn->setPrecomputedDistances(*distanceMatrix);
                    } else if(metric == "Pearson") {
                        auto knn = this->castToKNN<metrics::dist::Pearson>(learner);
                        knn->setPrecomputedDistances(*distanceMatrix);
                    } else if(metric == "KullbackLeibler") {
                        auto knn = this->castToKNN<metrics::dist::KullbackLeibler>(learner);
                        knn->setPrecomputedDistances(*distanceMatrix);
                    } else if(metric == "Hassanat") {
                        auto knn = this->castToKNN<metrics::dist::Hassanat>(learner);
                        knn->setPrecomputedDistances(*distanceMatrix);
                    }
                }
            }

            template<template<typename> class DistanceFunc>
            std::shared_ptr<classifier::KNNClassifier<T, DistanceFunc<T>>> castToKNN(std::shared_ptr<mltk::Learner<T>> learner) {
                return std::dynamic_pointer_cast<classifier::KNNClassifier<T, DistanceFunc<T>>>(learner);
            }
            
            template<template<typename> class DistanceFunc>
            std::shared_ptr<metrics::dist::BaseMatrix> computeDistanceMatrix(std::shared_ptr<mltk::Learner<T>> learner) {
                auto knnClassifier = castToKNN<DistanceFunc>(learner);
                if(knnClassifier != nullptr){
                    auto distances = knnClassifier->precomputeDistances(*this->samples);
                    return std::make_shared<metrics::dist::DistanceMatrix<DistanceFunc<T>>>(distances);
                }
                return nullptr;
            }

            std::map<std::string, std::shared_ptr<metrics::dist::BaseMatrix>> computeDistances() {
                std::map<std::string, std::shared_ptr<metrics::dist::BaseMatrix>> distance_matrices;
                    // Create metrics mapping
                std::map<std::string, std::function<std::shared_ptr<metrics::dist::BaseMatrix>(std::shared_ptr<mltk::Learner<T>>)>> metricsMap = {
                    {"Euclidean", [this](std::shared_ptr<mltk::Learner<T>> learner) { return this->computeDistanceMatrix<metrics::dist::Euclidean>(learner); }},
                    {"Lorentzian", [this](std::shared_ptr<mltk::Learner<T>> learner) { return this->computeDistanceMatrix<metrics::dist::Lorentzian>(learner); }},
                    {"Cosine", [this](std::shared_ptr<mltk::Learner<T>> learner) { return this->computeDistanceMatrix<metrics::dist::Cosine>(learner); }},
                    {"Bhattacharyya", [this](std::shared_ptr<mltk::Learner<T>> learner) { return this->computeDistanceMatrix<metrics::dist::Bhattacharyya>(learner); }},
                    {"Pearson", [this](std::shared_ptr<mltk::Learner<T>> learner) { return this->computeDistanceMatrix<metrics::dist::Pearson>(learner); }},
                    {"KullbackLeibler", [this](std::shared_ptr<mltk::Learner<T>> learner) { return this->computeDistanceMatrix<metrics::dist::KullbackLeibler>(learner); }},
                    {"Hassanat", [this](std::shared_ptr<mltk::Learner<T>> learner) { return this->computeDistanceMatrix<metrics::dist::Hassanat>(learner); }},
                };


                for(size_t i = 0; i < this->m_learners.size(); i++) {
                    auto learner = this->m_learners[i];
                    auto metric = this->metrics[i];

                    auto it = metricsMap.find(metric);
                    if (it != metricsMap.end()) {
                        auto matrix = it->second(learner);
                        distance_matrices[metric] = matrix;
                    }
                }
                
                return distance_matrices;
            }


            double maxAccuracy(){
                std::vector<int> ids(this->samples->size(), 0);

                int i = 0;
                for(auto point: this->samples->getPoints()) {
                    for (auto& learner: this->m_learners) {
                        if(learner->evaluate(*point) == point->Y()){
                            ids[i] = 1;
                            break;
                        }
                    }
                    i++;
                }
                double sum = std::accumulate(ids.begin(), ids.end(), 0.0);
                return sum/this->samples->size();
            }

            void setWeights(const mltk::Point<double> weights) {
                assert(weights.size() == this->m_learners.size());
                this->weights = weights;
            }

            mltk::Point<double> individualAccuracies() {
                mltk::Point<double> accs(this->m_learners.size(), 0.0);
                std::transform(this->m_learners.begin(), this->m_learners.end(), accs.begin(),
                                [this](auto learner) { 
                                    learner->setSamples(this->samples);
                                    learner->train();
                                    auto classifier = dynamic_cast<classifier::Classifier<T> *>(learner.get());
                                    auto acc = validation::kkfold(*this->samples, *classifier, 10, 10).accuracy/100.0;
                                    return acc;
                                }
                );

                return accs;
            }

            std::vector<std::string> metricsNames() {
                return this->metrics;
            }

            mltk::Point<double> getWeights(){ return this->weights; }

            double evaluate(const Point<double> &p, const std::string &metric) {
                auto it = std::find(this->metrics.begin(), this->metrics.end(), metric);

                if (it != this->metrics.end()) {
                    auto idx = it - this->metrics.begin();
                    auto learner = this->m_learners[idx];
                    learner->setSamples(this->samples);
                    return learner->evaluate(p);
                }
                return 0.0;
            }

            double evaluate(const Point<double> &p, bool raw_value = false) override {
                auto _classes = this->samples->classes();
                mltk::Point<double> votes(_classes.size(), 0.0);


                if (voting_type == "soft") {
                    assert(this->weights.size() > 0);
                } else {
                    this->weights = 1;
                }

                auto loop_body = [this, &votes, &p, &_classes](const int a, const int b){
                    for(int i = a; i < b; i++){
                        if(this->weights[i] == 0) continue;
                        this->m_learners[i]->setSamples(this->samples);
                        auto pred = this->m_learners[i]->evaluate(p);
                        auto comp = [&pred](const auto &a) {
                                        return (a == pred);
                                    };
                        size_t pred_pos = std::find_if(_classes.begin(), _classes.end(), comp) - _classes.begin();
                        votes[pred_pos] += this->weights[i]*this->m_learners[i]->getPredictionProbability();
                    }
                };

                pool.parallelize_loop(0, this->m_learners.size(), loop_body, this->m_learners.size());
                
                size_t max_votes = std::max_element(votes.X().begin(), votes.X().end()) - votes.X().begin();
                return _classes[max_votes];
            }

            std::string getFormulationString() override {
                return this->m_learners[0]->getFormulationString();
            }
        };
    }
}

#endif //UFJF_MLTK_KNNENSEMBLEW_HPP
