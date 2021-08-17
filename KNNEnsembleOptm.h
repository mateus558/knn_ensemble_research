//
// Created by mateus on 11/05/2021.
//

#ifndef KNN_RESEARCH_KNNENSEMBLEOPTM_H
#define KNN_RESEARCH_KNNENSEMBLEOPTM_H

#ifdef THREADS_ENABLED
#include <future>
#include <execution>
#endif
#include <ufjfmltk/ufjfmltk.hpp>
#include "alglib-3.17.0/optimization.h"
#include <armadillo>
#include <iomanip>


namespace mltk::ensemble{

    template<typename T>
class KNNEnsembleOptm : public Ensemble<T>, public classifier::Classifier<T> {
private:
    using matrix = arma::mat;
    using vec = arma::colvec;

    mltk::Point<double> start_point;
    bool has_startpoint{false}, mult_accs{false};
    double mse{0.0};

private:
    size_t folds{10};
    std::map<int, int> class_maper;
    mltk::Point<double> weights, accs;
public:
    [[nodiscard]] const Point<double> &getWeights() const;

    [[nodiscard]] const Point<double> &getAccs() const;

    double getMse() const;

    void setStartingPoint(const mltk::Point<double>& starting_point);

private:

    std::string vectorToAlglib(const arma::rowvec& vec);
    std::string matrixToAlglib(const matrix& mat);
    std::string pointToAlglib(const Point<double>& point);
    arma::rowvec findWeights(const matrix& Yhatj, const matrix& Y, size_t n_learners, int verbose);

public:
    KNNEnsembleOptm() = default;
    KNNEnsembleOptm(const Data<T> &data, size_t k, bool mult_accs = false, size_t folds=10, size_t seed=42, int verbose = 0):
    folds(folds), mult_accs(mult_accs) {
        this->samples = mltk::make_data<T>(data);
        this->seed = seed;
        this->m_learners.resize(7);
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
    std::string KNNEnsembleOptm<T>::pointToAlglib(const Point<double> &point) {
        std::string str_vec = "[";

        for(auto it = point.begin(); it != (point.end()-1); it++){
            str_vec += std::to_string(*it) + ",";
        }
        str_vec += std::to_string(point[point.size()-1]) + "]";
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
        try{
            matrix A = Yhat.t() * Yhat;
            arma::rowvec b = -Y.t() * Yhat;
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
            // set initial solution
            alglib::real_1d_array w;
            alglib::minqpstate state;
            alglib::minqpreport rep;

            alglib::minqpcreate(n_learners, state);
            alglib::minqpsetquadraticterm(state, A_);
            alglib::minqpsetlinearterm(state, b_);
            if(has_startpoint) {
                alglib::real_1d_array x0 = pointToAlglib(start_point).c_str();
                alglib::minqpsetstartingpoint(state, x0);
            }
            alglib::minqpsetbc(state, bndl, bndu);
            alglib::minqpsetlc(state, C_, ct);

            alglib::minqpsetscaleautodiag(state);
            //alglib::minqpsetalgodenseaul(state, 1.0e-9, 1.0e+4, 12);
            alglib::minqpsetalgodenseaul(state, 1.0e-9, 1.0e+4, 5);
            alglib::minqpoptimize(state);
            alglib::minqpresults(state, w, rep);
            arma::rowvec W = arma::rowvec(w.getcontent(), n_learners);
            if(verbose > 1) {
                std::cout << "Yhat: " << Yhat.n_rows << "x" << Yhat.n_cols << std::endl;
                std::cout << "Y: " << Y.n_rows << "x" << Y.n_cols << std::endl;
                std::cout << "A: " << A.n_rows << "x" << A.n_cols << std::endl;
                std::cout << "b: " << b.n_rows << "x" << b.n_cols << std::endl;
                std::cout << "C: " << C.n_rows << "x" << C.n_cols << std::endl;
                std::cout << "Termination type: " << rep.terminationtype << std::endl;
                ((Y - Yhat * W.t()).t() * (Y - Yhat * W.t())).print("MSE: ");
            }
            mse = as_scalar((Y - Yhat * W.t()).t() * (Y - Yhat * W.t()));
            return W;
        }catch(alglib::ap_error e) {
            std::cout << "error msg: " << e.msg << std::endl;
            exit(0);
        }
    }

    template<typename T>
    bool KNNEnsembleOptm<T>::train() {
        if(!this->weights.empty()) return true;
        size_t n_learners = this->m_learners.size();
        auto kfold_splits = mltk::validation::kfoldsplit(*this->samples, folds, false, this->seed);
#ifdef THREADS_ENABLED
        std::vector<std::pair<size_t, arma::colvec>> results(n_learners);
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
            class_maper[classes[0]] = 1;
            class_maper[classes[1]] = -1;
        }else{
#pragma unroll
            for(int i = 0; i < classes.size(); i++){
                class_maper[classes[i]] = classes[i];
            }
        }
        accs.resize(n_learners);
#pragma unroll
        for (size_t j = 0; j < n_learners; j++) {
            auto classifier = dynamic_cast<classifier::Classifier<T> *>(this->m_learners[j].get());
            auto report = validation::kkfold(*this->samples, *classifier, 10, 10, this->seed, 0);
            accs[j] = report.accuracy / 100.0;
        }
        for(size_t i = 0; i < kfold_splits.size(); i++) {
            auto train = kfold_splits[i].train;
            auto test = kfold_splits[i].test;
            arma::colvec Yj = arma::colvec(test.size());
            matrix Yhatj = matrix(test.size(), n_learners);

            if(this->verbose > 1) std::cout << "Fold " << i+1 << std::endl;

            for (int j = 0; j < test.size(); j++) {
                Yj(j) = class_maper[test(j).Y()];
            }
            Y = arma::join_cols(Y, Yj);
#ifdef THREADS_ENABLED
            std::map<int, int> mapper = this->class_maper;
            std::transform(std::execution::par, this->m_learners.begin(), this->m_learners.end(), results.begin(),
                           [mapper, train, test] (auto& learner){
                arma::colvec preds(test.size(), arma::fill::zeros);

                learner->setSamples(train);
                learner->train();

                for(int i = 0; i < test.size(); i++){
                    preds(i) = mapper.at(learner->evaluate(test(i)));
                }
                return std::make_pair(0, preds);
            });
#else
            std::transform(this->m_learners.begin(), this->m_learners.end(), results.begin(), [this, train, test]
                    (auto& learner){
                auto make_predictions = [&train, &test](LearnerPointer<T> learner, size_t learner_pos,
                                                        std::map<int, int> class_maper){
                    arma::colvec preds(test.size(), arma::fill::zeros);

                    learner->setSamples(train);
                    learner->train();

                    for(int i = 0; i < test.size(); i++){
                        preds(i) = class_maper[learner->evaluate(test(i))];
                        //preds(i) = learner->evaluate(test(i));
                    }
                    return std::make_pair(learner_pos, preds);
                };
                return make_predictions(learner, 0, class_maper);
            });
#endif
            matrix Y_learners = matrix(n_learners, kfold_splits[i].test.size());
            for (size_t col = 0; col < results.size(); col++) {
                Yhatj.col(col) = results[col].second*this->accs[col];
            }
            Yhat = arma::join_cols(Yhat, Yhatj);
        }
        if(this->verbose > 1) std::cout << "\nOptimizing weights\n" << std::endl;
        w = findWeights(Yhat, Y,n_learners, this->verbose);

        this->weights.resize(n_learners);
        this->weights = arma::conv_to<std::vector<double>>::from(w);
        if(this->verbose > 1) {
            std::cout << "Accuracies: " << accs << std::endl;
            std::cout << "Weights: " << this->weights << std::endl;
            std::cout << std::endl;
        }
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
            if(mult_accs) {
                votes[pred_pos] += std::abs(this->weights[i] * accs[i]*this->class_maper[pred]);
                sum += this->weights[i] * accs[i];
            }else {
                votes[pred_pos] += std::abs(this->weights[i]*this->class_maper[pred]);
                sum += this->weights[i];
            }
        }
        //std::cout << votes << std::endl;
        size_t max_votes = std::max_element(votes.X().begin(), votes.X().end()) - votes.X().begin();
        return _classes[max_votes];
        //return (sum < 0)?-1:1;
        //return sum;
    }

    template<typename T>
    const Point<double> &KNNEnsembleOptm<T>::getWeights() const {
        return weights;
    }

    template<typename T>
    const Point<double> &KNNEnsembleOptm<T>::getAccs() const {
        return accs;
    }

    template<typename T>
    double KNNEnsembleOptm<T>::getMse() const {
        return mse;
    }

    template<typename T>
    void KNNEnsembleOptm<T>::setStartingPoint(const Point<double> &starting_point) {
        this->start_point = starting_point;
        has_startpoint = true;
    }
}

#endif //KNN_RESEARCH_KNNENSEMBLEOPTM_H
