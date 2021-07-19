//
// Created by mateus on 11/05/2021.
//

#ifndef KNN_RESEARCH_KNNENSEMBLEOPTM_H
#define KNN_RESEARCH_KNNENSEMBLEOPTM_H

#ifdef THREADS_ENABLED
#include <future>
#endif
#include <ufjfmltk/ufjfmltk.hpp>
#include "alglib/optimization.h"
#include <armadillo>
#include <iomanip>


namespace mltk::ensemble{

    template<typename T>
class KNNEnsembleOptm : public Ensemble<T>, public classifier::Classifier<T> {
private:
    using matrix = arma::mat;
    using vec = arma::colvec;

    size_t folds{10};
    std::map<int, int> class_maper;
    mltk::Point<double> weights, accs;

    static std::string vectorToAlglib(const arma::rowvec& vec);
    static std::string matrixToAlglib(const matrix& mat);
    static arma::rowvec findWeights(const matrix& Yhatj, const matrix& Y, const size_t n_learners, int verbose);

public:
    KNNEnsembleOptm() = default;
    KNNEnsembleOptm(Data<T> &data, size_t k, size_t folds=10, size_t seed=42): folds(folds) {
        this->samples = mltk::make_data<T>(data);
        this->seed = seed;
        this->m_learners.resize(7);
//        this->m_learners[0] = std::make_shared<regressor::KNNRegressor<T, metrics::dist::Euclidean<T>>>(k);
//        this->m_learners[1] = std::make_shared<regressor::KNNRegressor<T, metrics::dist::Lorentzian<T>>>(k);
//        this->m_learners[2] = std::make_shared<regressor::KNNRegressor<T, metrics::dist::Cosine<T>>>(k);
//        this->m_learners[3] = std::make_shared<regressor::KNNRegressor<T, metrics::dist::Bhattacharyya<T>>>(k);
//        this->m_learners[4] = std::make_shared<regressor::KNNRegressor<T, metrics::dist::Pearson<T>>>(k);
//        this->m_learners[5] = std::make_shared<regressor::KNNRegressor<T, metrics::dist::KullbackLeibler<T>>>(k);
//        this->m_learners[6] = std::make_shared<regressor::KNNRegressor<T, metrics::dist::Hassanat<T>>>(k);
        this->m_learners[0] = std::make_shared<classifier::KNNClassifier<T, metrics::dist::Euclidean<T>>>(k);
        this->m_learners[1] = std::make_shared<classifier::KNNClassifier<T, metrics::dist::Lorentzian<T>>>(k);
        this->m_learners[2] = std::make_shared<classifier::KNNClassifier<T, metrics::dist::Cosine<T>>>(k);
        this->m_learners[3] = std::make_shared<classifier::KNNClassifier<T, metrics::dist::Bhattacharyya<T>>>(k);
        this->m_learners[4] = std::make_shared<classifier::KNNClassifier<T, metrics::dist::Pearson<T>>>(k);
        this->m_learners[5] = std::make_shared<classifier::KNNClassifier<T, metrics::dist::KullbackLeibler<T>>>(k);
        this->m_learners[6] = std::make_shared<classifier::KNNClassifier<T, metrics::dist::Hassanat<T>>>(k);
    }

    bool train() override;

    double evaluate(const Point<T> &p, bool raw_value = false) override;

    std::string getFormulationString() override{
        return "Primal";
    }
};

    template<typename T>
    std::string KNNEnsembleOptm<T>::vectorToAlglib(const arma::rowvec& vec) {
        std::string str_vec = "[";

        for(auto it = vec.begin(); it != (vec.end()-1); it++){
            str_vec += std::to_string(*it) + ",";
        }
        str_vec += std::to_string(vec.back()) + "]";
        return str_vec;
    }

    template<typename T>
    std::string KNNEnsembleOptm<T>::matrixToAlglib(const matrix &mat) {
        std::string str_mat = "[";
        int i;
        for(i = 0; i < mat.n_rows-1; i++){
            str_mat += vectorToAlglib(mat.row(i)) + ",";
        }
        str_mat += vectorToAlglib(mat.row(i)) + "]";
        return str_mat;
    }

    template<typename T>
    arma::rowvec KNNEnsembleOptm<T>::findWeights(const KNNEnsembleOptm::matrix &Yhat,
                                                                     const KNNEnsembleOptm::matrix &Y,
                                                                     const size_t n_learners, int verbose) {
        matrix A = 2.0 * Yhat.t() * Yhat;
        arma::rowvec b = -0.5 * Y.t() * Yhat;
        matrix C = arma::ones(1,n_learners+1);
        arma::rowvec lower_bound = arma::rowvec(n_learners, arma::fill::zeros);
        arma::rowvec upper_bound = arma::rowvec(n_learners, arma::fill::ones);
        arma::rowvec scale = arma::rowvec(n_learners, arma::fill::ones);

        alglib::real_2d_array A_ = matrixToAlglib(A).c_str();
        alglib::real_1d_array b_ = vectorToAlglib(b).c_str();
        alglib::real_2d_array C_ = matrixToAlglib(C).c_str();
        alglib::integer_1d_array ct = "[0]";
        alglib::real_1d_array s = vectorToAlglib(scale).c_str();
        alglib::real_1d_array bndl = vectorToAlglib(lower_bound).c_str();
        alglib::real_1d_array bndu = vectorToAlglib(upper_bound).c_str();
        alglib::real_1d_array w;
        alglib::minqpstate state;
        alglib::minqpreport rep;

        alglib::minqpcreate(n_learners, state);
        alglib::minqpsetquadraticterm(state, A_);
        alglib::minqpsetlinearterm(state, b_);
        alglib::minqpsetbc(state, bndl, bndu);
        alglib::minqpsetlc(state, C_, ct);

        alglib::minqpsetscaleautodiag(state);
        alglib::minqpsetalgodenseaul(state, 1.0e-9, 1.0e+4, 12);
        alglib::minqpoptimize(state);
        alglib::minqpresults(state, w, rep);
        arma::rowvec W = arma::rowvec(w.getcontent(), n_learners);
        if(verbose) {
            std::cout << "Yhat: " << Yhat.n_rows << "x" << Yhat.n_cols << std::endl;
            std::cout << "Y: " << Y.n_rows << "x" << Y.n_cols << std::endl;
            std::cout << "A: " << A.n_rows << "x" << A.n_cols << std::endl;
            std::cout << "b: " << b.n_rows << "x" << b.n_cols << std::endl;
            std::cout << "C: " << C.n_rows << "x" << C.n_cols << std::endl;
            std::cout << "Termination type: " << rep.terminationtype << std::endl;
            ((Y - Yhat * W.t()).t() * (Y - Yhat * W.t())).print("MSE: ");
        }
        return W;
    }

    template<typename T>
    bool KNNEnsembleOptm<T>::train() {
        size_t n_learners = this->m_learners.size();
        auto kfold_splits = mltk::validation::kfoldsplit(*this->samples, folds, false, this->seed);
#ifdef THREADS_ENABLED
        std::vector<std::future<std::pair<size_t, arma::colvec>>> results(n_learners);
#else
        std::vector<std::pair<size_t, arma::colvec>> results(n_learners);
#endif
        arma::rowvec w = arma::rowvec(this->m_learners.size(), arma::fill::randu);
        arma::colvec Y;
        matrix Yhat;
        auto classes = this->samples->classes();
        std::cout << std::fixed << std::showpoint;
        std::cout << std::setprecision(3);
        if(classes.size() == 2){
            class_maper[classes[0]] = classes[0];
            class_maper[classes[1]] = classes[1];
        }else{
            for(int i = 0; i < classes.size(); i++){
                class_maper[classes[i]] = classes[i];
            }
        }

        accs.resize(n_learners);
        for(size_t j = 0; j < n_learners; j++){
            auto classifier = dynamic_cast<classifier::Classifier<T> *>(this->m_learners[j].get());
            auto report = validation::kkfold(*this->samples, *classifier, 10, 10, this->seed, 0);
            accs[j] = report.accuracy/100.0;
        }
        for(size_t i = 0; i < kfold_splits.size(); i++) {
            auto train = kfold_splits[i].train;
            auto test = kfold_splits[i].test;
            arma::colvec Yj = arma::colvec(test.size());
            matrix Yhatj = matrix(test.size(), n_learners);

            if(this->verbose) std::cout << "Fold " << i+1 << std::endl;

            for (int j = 0; j < test.size(); j++) {
                Yj(j) = class_maper[test(j).Y()];
//                Yj(j) = test(j).Y();
            }

            Y = arma::join_cols(Y, Yj);
            for (size_t j = 0; j < n_learners; j++) {
                auto make_predictions = [&train, &test](LearnerPointer<T> learner, size_t learner_pos,
                        std::map<int, int> class_maper){
                    arma::colvec preds(test.size(), arma::fill::zeros);

                    learner->setSamples(train);
                    learner->train();

                    for(int i = 0; i < test.size(); i++){
//                        preds(i) = class_maper[learner->evaluate(test(i))];
                        preds(i) = learner->evaluate(test(i));
                    }
                    return std::make_pair(learner_pos, preds);
                };
#ifdef THREADS_ENABLED
                    results[j] = std::async(std::launch::async, make_predictions, this->m_learners[j], j, class_maper);
#else
                    results[j] = make_predictions(this->m_learners[j], j, class_maper);
#endif
            }

            matrix Y_learners = matrix(n_learners, kfold_splits[i].test.size());
            for (auto &res: results) {
#ifdef THREADS_ENABLED
                auto result = res.get();
                Yhatj.col(result.first) = result.second;
#else
                Yhatj.col(res.first) = res.second;
#endif
            }
            Yhat = arma::join_cols(Yhat, Yhatj);
        }
       // Yhat.print();
        if(this->verbose) std::cout << "\nOptimizing weights\n" << std::endl;
        w = findWeights(Yhat, Y,n_learners, this->verbose);
        std::cout << accs << std::endl;
        this->weights.resize(n_learners);
        this->weights = arma::conv_to<std::vector<double>>::from(w);
        std::cout << this->weights << std::endl;
        this->weights = (1.0-this->weights);
        std::cout << this->weights << std::endl;

        return true;
    }

    template<typename T>
    double KNNEnsembleOptm<T>::evaluate(const Point<T> &p, bool raw_value) {
        auto _classes = this->samples->classes();
        auto sum = 0.0;
        mltk::Point<double> votes(_classes.size(), 0.0);

        for (size_t i = 0; i < this->m_learners.size(); i++) {
            if(this->weights[i] == 0) continue;
            auto pred = this->m_learners[i]->evaluate(p);
            // get prediction position
            size_t pred_pos = std::find_if(_classes.begin(), _classes.end(), [&pred](const auto &a) {
                return (a == pred);
            }) - _classes.begin();
            // count prediction as a vote
            votes[pred_pos] += std::abs(this->weights[i]*class_maper[pred]);
            sum += this->weights[i]*class_maper[pred];
        }
        //std::cout << votes << std::endl;
        size_t max_votes = std::max_element(votes.X().begin(), votes.X().end()) - votes.X().begin();
        return _classes[max_votes];
        return (sum < 0)?-1:1;
        //return sum;
    }
}

#endif //KNN_RESEARCH_KNNENSEMBLEOPTM_H
