//
// Created by Marim on 21/02/2022.
//

#ifndef KNN_RESEARCH_KNNENSEMBLELIN_HPP
#define KNN_RESEARCH_KNNENSEMBLELIN_HPP

#include <ufjfmltk/ufjfmltk.hpp>

namespace mltk::ensemble {

    template<typename T = double>
    class KNNEnsembleLin : public Ensemble<T>, public classifier::Classifier<T> {
    private:
        bool mult_accs{false};
        size_t folds{10};

    public:
        KNNEnsembleLin() = default;

        KNNEnsembleLin(const Data <T> &data, size_t k, bool mult_accs = false, size_t folds = 10,
                       size_t seed = 42, int verbose = 0) :
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

        double evaluate(const Point <T> &p, bool raw_value = false) override;

        std::string getFormulationString() override {
            return "Primal";
        }

        void set_samples(mltk::Data<T> &data) {
            for (auto &learner: this->m_learners) {
                learner->setSamples(data);
            }
        }
    };

    template<typename T>
    bool KNNEnsembleLin<T>::train() {
        size_t n_learners = this->m_learners.size();
        auto kfold_splits = mltk::validation::kfoldsplit(*this->samples, folds, true, this->seed);
        std::vector<std::vector<double>> C(n_learners);

        for(auto split: kfold_splits){
            auto train_data = split.train;
            auto test_data = split.test;

            for (size_t i = 0; i < n_learners; i++) {
                this->m_learners[i]->setSamples(train_data);
                this->m_learners[i]->train();
            }
            for (size_t i = 0; i < n_learners; i++) {
                for (auto &test_point: test_data) {
                    this->m_learners[i]->evaluate(*test_point);
                    double prob = this->m_learners[i]->getPredictionProbability();
                    C[i].push_back(prob);
                }
            }
        }
        for(int i = 0; i < C.size(); i++){
            for(int j = 0; j < C[i].size(); j++){
                std::cout << C[i][j] << " ";
            }
            std::cout << std::endl;
        }

        return false;
    }

    template<typename T>
    double KNNEnsembleLin<T>::evaluate(const Point <T> &p, bool raw_value) {
        return 0;
    }

}
#endif //KNN_RESEARCH_KNNENSEMBLELIN_HPP
