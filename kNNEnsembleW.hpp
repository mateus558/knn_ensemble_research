//
// Created by mateuscmarim on 05/02/2021.
//

#ifndef UFJF_MLTK_KNNENSEMBLEW_HPP
#define UFJF_MLTK_KNNENSEMBLEW_HPP

#include <ufjfmltk/ufjfmltk.hpp>
#include "thread_pool.hpp"


namespace mltk {
    namespace ensemble {
        template<typename T>
        class kNNEnsembleW: public Ensemble<T>, public classifier::Classifier<T> {
        private:
            size_t k;
            std::string voting_type = "soft";
            mltk::Point<double> weights;
            thread_pool pool{16};
            std::vector<std::string> metrics;

        public:
            kNNEnsembleW() = default;
            kNNEnsembleW(Data<T> &samples, size_t _k): k(_k) {
                this->samples = make_data<T>(samples);

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
                // this->m_learners.push_back(std::make_shared<classifier::KNNClassifier<T, metrics::dist::Hassanat<T>>>(k));
                // metrics.push_back("Hassanat");

                std::vector<double> w;
                for (size_t i = 0; i < this->m_learners.size(); i++) {
                    this->m_learners[i]->setSamples(this->samples);
                    this->m_learners[i]->train();
                  //  auto classifier = dynamic_cast<classifier::Classifier<T> *>(this->m_learners[i].get());
                  //  auto acc = validation::kkfold(samples, *classifier, 10, 10, this->seed, 0).accuracy/100.0;
                    //w.push_back(acc);
                }
                //this->weights = w;
            }

            bool train() override{

                return true;
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
                for (size_t i = 0; i < this->m_learners.size(); i++) {
                    this->m_learners[i]->setSamples(this->samples);
                    this->m_learners[i]->train();
                    auto classifier = dynamic_cast<classifier::Classifier<T> *>(this->m_learners[i].get());
                    auto acc = validation::kkfold(*this->samples, *classifier, 10, 10).accuracy/100.0;
                    accs[i] = acc;
                }

                return accs;
            }

            std::vector<std::string> metricsNames() {
                return this->metrics;
            }

            mltk::Point<double> getWeights(){ return this->weights; }

            double evaluate(const Point<T> &p, bool raw_value = false) override {
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
                        auto pred = this->m_learners[i]->evaluate(p);
                        auto comp = [&pred](const auto &a) {
                                        return (a == pred);
                                    };
                        size_t pred_pos = std::find_if(_classes.begin(), _classes.end(), comp) - _classes.begin();
                        votes[pred_pos] += this->weights[i]*this->m_learners[i]->getPredictionProbability();
                    }
                };

                pool.parallelize_loop(0, this->m_learners.size()-1, loop_body, this->m_learners.size());

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
