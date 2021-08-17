//
// Created by mateuscmarim on 05/02/2021.
//

#ifndef UFJF_MLTK_KNNENSEMBLEW_HPP
#define UFJF_MLTK_KNNENSEMBLEW_HPP

#include <ufjfmltk/ufjfmltk.hpp>
#include <future>
#include <iomanip>
#include "utils.h"

namespace mltk {
    namespace ensemble {
        template<typename T>
        class kNNEnsembleWSS: public Ensemble<T>, public classifier::Classifier<T> {
        private:
            size_t k;
            std::string voting_type = "soft";
            mltk::Point<double> weights;
            mltk::Point<double> sub_weights;
        public:
            kNNEnsembleWSS() = default;
            kNNEnsembleWSS(Data<T> &samples, size_t _k): k(_k) {
                this->samples = make_data<T>(samples);
                this->m_learners.resize(7);
                this->m_learners[0] = std::make_shared<classifier::KNNClassifier<T, metrics::dist::Euclidean<T>>>(k);
                this->m_learners[1] = std::make_shared<classifier::KNNClassifier<T, metrics::dist::Lorentzian<T>>>(k);
                this->m_learners[2] = std::make_shared<classifier::KNNClassifier<T, metrics::dist::Cosine<T>>>(k);
                this->m_learners[3] = std::make_shared<classifier::KNNClassifier<T, metrics::dist::Bhattacharyya<T>>>(k);
                this->m_learners[4] = std::make_shared<classifier::KNNClassifier<T, metrics::dist::Pearson<T>>>(k);
                this->m_learners[5] = std::make_shared<classifier::KNNClassifier<T, metrics::dist::KullbackLeibler<T>>>(k);
                this->m_learners[6] = std::make_shared<classifier::KNNClassifier<T, metrics::dist::Hassanat<T>>>(k);

//                std::vector<double> w;
//                for (size_t i = 0; i < this->m_learners.size(); i++) {
//                    this->m_learners[i]->setSamples(this->samples);
//                    this->m_learners[i]->train();
//                    auto classifier = dynamic_cast<classifier::Classifier<T> *>(this->m_learners[i].get());
//                    auto acc = validation::kkfold(samples, *classifier, 10, 10, this->seed, 0).accuracy/100.0;
//                    w.push_back(acc);
//                }
//                this->weights = w;
            }

            std::vector<double> linspace(double lower, double upper, size_t N){
                double h = (upper - lower) / static_cast<double>(N-1);
                std::vector<double> xs(N);
                std::vector<double>::iterator x;
                double val;
                for (x = xs.begin(), val = lower; x != xs.end(); ++x, val += h) {
                    *x = val;
                }
                return xs;
            }

            void optimize(Data<T> &samples, classifier::Classifier<T> &classifier, std::vector<double>& set, std::vector<double>& perm,
                          int _k, double total_sum, Point<double>& best, double& best_acc, int& N){
                std::string log_fname = samples.name() + ".log", csv_fname = samples.name() + ".csv";

                auto sum = std::accumulate(perm.begin(), perm.end(), 0.0);

                if(sum > total_sum){
                    return;
                }

                if(_k == 0){
                    Point<double> sub_w(perm);

                   if(sum == total_sum){
                        this->sub_weights = sub_w;
                        auto acc = validation::kkfold(samples, classifier, 10, 10, this->seed, 0).accuracy;
                        if(acc > best_acc){
                            best_acc = acc;
                            best = perm;
                            mutex.lock();
                            std::cout << samples.name() << "  " << k << "  " << N << "    " << acc << "   " << best <<
                            "   " << best << "     " << best_acc << std::endl;
                            log_files[log_fname] << samples.name() << "  " << k << "  " << N << "    " << acc << "   " << best <<
                            "   " << best << "     " << best_acc << std::endl;
                            log_files[csv_fname] << samples.name() << ";" << k << ";" << N << ";" << acc << ";" << best <<
                            ";" << best << ";" << best_acc << std::endl;
                            mutex.unlock();
                        }else if(N%50 == 0 && N > 0){
                            Point<double> w(perm);
                            mutex.lock();
                            std::cout << samples.name() << "  " << k << "  " << N << "    " << acc << "   " << w <<
                            "   " << best << "     " << best_acc << std::endl;
                            log_files[log_fname] << samples.name() << "  " << k << "  " << N << "    " << acc << "   " << w <<
                            "   " << best << "     " << best_acc << std::endl;
                            log_files[csv_fname] << samples.name() << ";" << k << ";" << N << ";" << acc << ";" << w <<
                            ";" << best << ";" << best_acc << std::endl;
                            mutex.unlock();
                        }
                        N++;
                    }
                    perm.clear();
                    return;
                }

                for (int i = 0; i < set.size(); i++){
                    std::vector<double> p = perm;
                    p.push_back(set[i]);

                    optimize(samples, classifier,set, p, _k - 1, total_sum, best, best_acc,N);
                }
            }

            void optimizeSubWeights(Data<T> &samples, size_t step, double max_weight){
                Point<double> temp(this->m_learners.size(), 0), best(this->m_learners.size(), 0);
                this->sub_weights.resize(this->m_learners.size());
                std::vector<double> values = linspace(0, max_weight, step);
                Point<double> p(values);
                double best_acc = 0.0;
                int N = 0, _k = this->m_learners.size();
                std::vector<double> perm;

                optimize(samples, *this, values, perm, _k, max_weight, best, best_acc, N);
                this->sub_weights = best;
            }

            Point<double> getSubWeights(){
                return this->sub_weights;
            }

            Point<double> getWeights(){
                return this->weights;
            }

            bool train() override{
                if(!this->weights.empty()) return true;
                std::vector<std::pair<mltk::LearnerPointer<T>, mltk::Data<T>>> par_accs_exec;
                for(size_t i = 0; i < this->m_learners.size(); i++){
                    par_accs_exec.emplace_back(this->m_learners[i], this->samples->copy());
                }
                std::vector<double> w;
                std::vector<std::future<double>> futures(par_accs_exec.size());

                auto acc_estimator = [](auto estimator, size_t seed){
                    auto learner = estimator.first;
                    learner->setSamples(estimator.second);
                    learner->train();
                    auto classifier = dynamic_cast<classifier::Classifier<T> *>(learner.get());
                    return validation::kkfold(estimator.second, *classifier, 10, 10, seed, 0).accuracy/100.0;
                };
                for(size_t i = 0; i < futures.size(); i++){
                    futures[i] = std::async(std::launch::async, acc_estimator, par_accs_exec[i], 0);
                }
                for(auto& future: futures){
                    w.push_back(future.get());
                }
                this->weights = w;
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
                return sum/ this->samples->size();
            }

            void setWeights(const std::vector<double> weights) {
                assert(weights.size() == this->m_learners.size());
                this->weights.X().resize(weights.size());
                this->weights = weights;
            }

            double evaluate(const Point<T> &p, bool raw_value = false) override {
                auto _classes = this->samples->classes();
                mltk::Point<double> votes(_classes.size(), 0.0);

                if (voting_type == "soft") {
                    assert(this->weights.size() > 0);
                } else {
                    this->weights = 1;
                }
                for (size_t i = 0; i < this->m_learners.size(); i++) {
                    if(this->sub_weights[i] == 0) continue;
                    auto pred = this->m_learners[i]->evaluate(p);
                    // get prediction position
                    size_t pred_pos = std::find_if(_classes.begin(), _classes.end(), [&pred](const auto &a) {
                        return (a == pred);
                    }) - _classes.begin();
                    // count prediction as a vote
                    votes[pred_pos] += this->weights[i]*this->sub_weights[i];
                }
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
