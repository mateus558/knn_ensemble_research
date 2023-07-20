#include "SimulatedAnnealing1.hpp"
#include "kNNEnsembleW.hpp"
#include <random>
#include <cmath>
#include <nlohmann/json.hpp>
#include <limits>


namespace mltk {

    SimulatedAnnealing::SimulatedAnnealing(const mltk::Data<double> &data, size_t k, double T, double alpha, size_t minTempIter){
        this->samples = data;
        this->T = T;
        this->alpha = alpha;
        this->k = k;
        this->minTempIter = minTempIter;
    }

    mltk::Point<double> SimulatedAnnealing::neighbour(mltk::Point<double> w){
        std::random_device rd;
        std::mt19937 mt(rd());
        std::uniform_int_distribution<> int_dis(0, w.size() - 1);
        std::uniform_real_distribution<double> real_dis(-0.1, 0.1);

        size_t i = int_dis(mt);
        size_t j = int_dis(mt);

        // Ensure that i != j
        for(; j == i; j = int_dis(mt));

        double delta = real_dis(mt);

         // Ensure that weights[i] + delta is within the range [0, 1]
        delta = std::min(std::max(delta, -w[i]), 1.0 - w[i]);
        w[i] = w[i] + delta;
        
        // Ensure that weights[j] - delta is within the range [0, 1]
        delta = std::min(std::max(delta, w[j] - 1.0), w[j]);
        w[j] = w[j] - delta;

        // Adjust weights[j] so that sum(weights) = 1
        double total = w.sum() - w[j];

        // Ensure that total is not more than 1 (normalization)
        if (total > 1) {
            for(int k = 0; k < w.size(); k++){
                if(k != j) {
                    w[k] = w[k] / total;
                }
            }
            total = 1;
        }

        w[j] = 1 - total;

        return w;
    }

    bool accept(double current_cost, double new_cost, double T) {
        // If the new solution is better, accept it
        if (new_cost < current_cost) {
            return true;
        } else {             // If the new solution is worse, accept it with a probability according to the Boltzmann distribution
            std::random_device rd;
            std::mt19937 mt(rd());
            std::uniform_real_distribution<double> prob(0, 1);
            double delta = (new_cost - current_cost)*100;
            double P = std::exp( -delta / T);
            double r = prob(mt);  // Random number between 0 and 1

            if (r < P) {
                return true;
            } else {
                return false;
            }
        }
    }

    std::pair<mltk::Point<double>, json> SimulatedAnnealing::optimize(){
        mltk::ensemble::kNNEnsembleW<double> knn_ensemb(this->samples, this->k);
        mltk::Point<double> accs = knn_ensemb.individualAccuracies();
        // the initial solution is a discrete uniform distribution
        mltk::Point<double> initial_solution(knn_ensemb.size(), 1.0/knn_ensemb.size());
        mltk::Point<double> current_solution = initial_solution;
        mltk::Point<double> best_solution = initial_solution;
        std::vector<double> temp_avgs;
        std::vector<std::pair<mltk::Point<double>, double>> accepted;
        
        double t = this->T;
        double best_eval = this->objective(initial_solution);
        double current_eval = best_eval;
        double new_eval = best_eval;
        double curr_avg = 0;
        double prev_avg = std::numeric_limits<double>::infinity();

        std::cout << "Dataset: " << this->samples.name() << std::endl;
        std::cout << "Metrics: " <<  mltk::Point<std::string>(knn_ensemb.metricsNames()) << std::endl;
        std::cout << "Individual accs: " << accs*100 << std::endl;
        std::cout << "Initial solution: " << initial_solution << " " << initial_solution.sum();
        std::cout << " eval: " << best_eval << std::endl;
        std::cout << "k: " << this->k << " T: " << this->T << " alpha: " << this->alpha << " minTempIter: " << this->minTempIter << std::endl;
        std::cout << std::endl;

        // global number of iterations
        size_t it;

        accepted.push_back(std::make_pair(current_solution, current_eval));
        
        for(it = 0; t >= this->minT && best_eval > 0; ){
        //for(it = 0; t >= this->minT && curr_avg < prev_avg && best_eval > 0; ){
            // at each temperature, we set the partial sum to the last accepted solution evaluation
            double partial_sum = accepted.back().second;
            size_t tit = 0;    // tit is the number of iterations on the current temperature
            double diff_avg = 1;

            // the main loop will run while the diference between the min and max avg of the last
            // minTempIter is greater than a threshould epslon
            while(diff_avg > epslon && best_eval > 0){
                it++;
                tit++;

                double min_avg = std::numeric_limits<double>::infinity();
                double max_avg = -std::numeric_limits<double>::infinity();
                Point<double> new_solution = this->neighbour(current_solution);
                new_eval = this->objective(new_solution);   

                if(accept(current_eval, new_eval, t)){
                    double old_eval = current_eval;
                    current_eval = new_eval;
                    current_solution = new_solution;

                    accepted.push_back(std::make_pair(current_solution, current_eval));
                   
                    if(current_eval < best_eval){
                        best_eval = current_eval;
                        best_solution = current_solution;
                        std::cout << "\n[" + this->samples.name() + "] solution: " << current_solution << " eval: " << current_eval << " old_eval: " << old_eval << " t: " << t << " curr_temp_avg: " << curr_avg << " diff_avg: " << diff_avg << std::endl;
                        std::cout << "it: " << it << " tit: " << tit << " best solution: " << best_solution << " best_eval: " << best_eval << " k=" << k << std::endl;
                    }

                }else{
                    accepted.push_back(std::make_pair(best_solution, best_eval));
                }
                
                // add the new solution eval to the partial sum
                partial_sum += accepted.back().second;
                // compute the current average
                curr_avg = partial_sum / tit;
                temp_avgs.push_back(curr_avg);

                // check if it reached the min number of iterations on the current temperature
                if(tit > this->minTempIter){
                    // compute the min and max average on the last minTempIter iterations
                    for(size_t i = temp_avgs.size()-1; i > (temp_avgs.size()-1-this->minTempIter); i--){
                        if(temp_avgs[i] < min_avg){
                            min_avg = temp_avgs[i];
                        }

                        if(temp_avgs[i] > max_avg){
                            max_avg = temp_avgs[i];
                        }
                    }
                    diff_avg = max_avg - min_avg;
                    prev_avg = curr_avg;
                }
            }
            // decrease the temperature by (1 - alpha) rate
            t *= this->alpha;
            std::cout << "\n[" + this->samples.name() + "] " << "it: " << it << " t: " << t << " alpha:" << alpha << " k=" << k << " tit: " << tit << std::endl;
        }

        std::vector<double> w = best_solution.X();
        
        json report;
        report["accepted_solutions"] = accepted;
        report["k"] = this->k;
        report["T"] = this->T;
        report["alpha"] = this->alpha;
        report["initial_solution"] = initial_solution.X();
        report["initial_eval"] = this->objective(initial_solution);
        report["stop_temperature"] = t;
        report["iterations"] = it;
        report["min_T"] = this->minT;
        report["averages"] = temp_avgs;
        report["weights"] = w;
        report["best_eval"] = best_eval;
        report["individual_accuracies"] = accs;
        report["metrics"] = knn_ensemb.metricsNames();
        
        std::cout << "Best solution: " << best_solution << std::endl;
        std::cout << "Best eval: " << best_eval << std::endl;
        std::cout << "Individual acuracies: " << accs*100 << std::endl;
    
        return std::make_pair(best_solution, report);
    }

    double SimulatedAnnealing::objective(mltk::Point<double> weights){
        thread_pool pool;
        Point<double> errors(10);
        auto k = this->k;
        std::vector<mltk::Data<double>> data(10);

        for(size_t i = 0; i < 10; i++){
            data[i] = this->samples.copy();
        }

        auto loop_body = [&data, &errors, weights, k] (const int a, const int b) -> void {
            for(size_t i = a; i < b; i++){
                std::random_device rd;
                std::mt19937 mt(rd());
                std::uniform_int_distribution<size_t> int_dis(0, std::numeric_limits<size_t>::max());
                mltk::ensemble::kNNEnsembleW<double> knn_ensemb(data[i], k);
                knn_ensemb.setWeights(weights.X());
                
                double acc = mltk::validation::kfold(data[i], knn_ensemb, 10, true, int_dis(mt)).accuracy/100;
                errors[i] += 1.0 - acc;
            }
        };

        pool.parallelize_loop(0, 9, loop_body, 10); 

        return errors.sum()/errors.size();
    }
};