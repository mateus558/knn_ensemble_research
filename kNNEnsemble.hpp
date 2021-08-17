//
// Created by mateuscmarim on 20/01/2021.
//

#ifndef UFJF_MLTK_KNNENSEMBLE_HPP
#define UFJF_MLTK_KNNENSEMBLE_HPP

#include <ufjfmltk/Classifier.hpp>
#include <ufjfmltk/Ensemble.hpp>

namespace mltk {
namespace ensemble {
        template<typename T>
        class kNNEnsemble : public Ensemble<T>, public classifier::Classifier<T> {
        private:
            size_t k;
            bool test_mode = false;
        public:
            kNNEnsemble() = default;
            kNNEnsemble(const Data<T> &samples, size_t _k): k(_k) {
                this->samples = make_data<T>(samples);
                this->m_learners.resize(7);
                this->m_learners[0] = std::make_shared<classifier::KNNClassifier<T, metrics::dist::Euclidean<T>>>(k);
                this->m_learners[1] = std::make_shared<classifier::KNNClassifier<T, metrics::dist::Lorentzian<T>>>(k);
                this->m_learners[2] = std::make_shared<classifier::KNNClassifier<T, metrics::dist::Cosine<T>>>(k);
                this->m_learners[3] = std::make_shared<classifier::KNNClassifier<T, metrics::dist::Bhattacharyya<T>>>(k);
                this->m_learners[4] = std::make_shared<classifier::KNNClassifier<T, metrics::dist::Pearson<T>>>(k);
                this->m_learners[5] = std::make_shared<classifier::KNNClassifier<T, metrics::dist::KullbackLeibler<T>>>(k);
                this->m_learners[6] = std::make_shared<classifier::KNNClassifier<T, metrics::dist::Hassanat<T>>>(k);

                for(auto& learner: this->m_learners){
                    learner->setSamples(this->samples);
                }
            }

            mltk::Point<double> individual_accs(){
                std::vector<double> accs;
                for (size_t i = 0; i < this->m_learners.size(); i++) {
                    this->m_learners[i]->setSamples(this->samples);
                    this->m_learners[i]->train();
                    auto classifier = dynamic_cast<classifier::Classifier<T> *>(this->m_learners[i].get());
                    auto acc = validation::kkfold(*this->samples, *classifier, 10, 10, this->seed, 0).accuracy;
                    accs.push_back(acc);
                }
                return accs;
            }

            bool train() override{
                return true;
            }

            void set_test(bool is_test){ this->test_mode = is_test; }

            double evaluate(const Point<T> &p, bool raw_value = false) override {
                auto classes = this->samples->classes();
                double res = -100;
                Point<int> votes(classes.size());
                for (size_t i = 0; i < this->m_learners.size(); i++) {
                    int pred = this->m_learners[i]->evaluate(p);
                    if(pred == p.Y()){
                        res = p.Y();
                        break;
                    }
                    size_t pred_pos = std::find(classes.begin(), classes.end(), pred) - classes.begin();
                    votes[pred_pos]++;
                }
                if (!test_mode) {
                    return classes[std::max_element(votes.X().begin(), votes.X().end()) - votes.X().begin()];
                }
                return res;
            }
            std::string getFormulationString() override {
                return this->m_learners[0]->getFormulationString();
            }
        };
    }
}
#endif //UFJF_MLTK_KNNENSEMBLE_HPP
