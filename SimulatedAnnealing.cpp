#include "SimulatedAnnealing.hpp"
#include "kNNEnsembleW.hpp"
#include <random>


namespace mltk {

    SimulatedAnnealing::SimulatedAnnealing(const mltk::Data<double> &data, size_t k, double max_weight, double step, double T, double alpha, size_t max_iterations){
        this->samples = data;
        this->max_iterations = max_iterations;
        this->T = T;
        this->alpha = alpha;
        this->k = k;
        this->searchSpace = generateSearchSpace(k, max_weight, step);
        std::random_device rd;
        std::mt19937 mt(rd());

        std::cout << "Sanity check" << std::endl;
        std::cout << "---------------------" << std::endl;
        std::cout << std::endl;
        std::cout << "Search space size: " << this->searchSpace.size() << "\n" << std::endl;

        //std::shuffle (this->searchSpace.begin(), this->searchSpace.end(), mt);


        for(int i = 0; i < 5; i++){
            std::cout << this->searchSpace[i] << " Sum: " << this->searchSpace[i].sum() <<  std::endl;
        }

        std::cout << "..." << std::endl;
        
        for(int i = this->searchSpace.size() - 5; i < this->searchSpace.size(); i++){
            std::cout << this->searchSpace[i] << " Sum: " << this->searchSpace[i].sum() <<  std::endl;
        }
    }

    std::pair<mltk::Point<double>, size_t> SimulatedAnnealing::next(int neighbors, int curr_pos){
        std::random_device rd;
        std::mt19937 mt(rd());

        //std::cout << curr_pos - neighbors << std::endl;
        if((curr_pos - neighbors) <= 0){
            std::uniform_int_distribution<> dis(curr_pos, neighbors + curr_pos);
            size_t pos = dis(mt);
            //std::cout << "Pos: " << pos << std::endl;
            return std::make_pair(this->searchSpace[pos], pos);
        } else if(curr_pos + neighbors >= this->searchSpace.size() - 1){
            std::uniform_int_distribution<> dis(this->searchSpace.size() - neighbors - 1, this->searchSpace.size() - 2);
            size_t pos = dis(mt);
            //std::cout << "Pos: " << pos << std::endl;
            return std::make_pair(this->searchSpace[pos], pos);
        } else {
            size_t end = curr_pos + neighbors;
            if(end >= this->searchSpace.size()){
                end = this->searchSpace.size() - 1;
            }
            std::uniform_int_distribution<> dis(curr_pos + 1, end);
            //std::uniform_int_distribution<> dis(curr_pos - neighbors/2 - 1, curr_pos + neighbors/2 + 1);
            //std::cout << "Curr pos: " << curr_pos << std::endl;
            //std::cout << "Dis: " << curr_pos - neighbors/2 - 1 << " " << curr_pos + neighbors/2 + 1 << std::endl;
            while(true){
                size_t pos = dis(mt);
                //std::cout << "Pos: " << pos << std::endl;

                if(pos != curr_pos){
                    return std::make_pair(this->searchSpace[pos], pos);
                }
            }
        }
        
    }

    Point<double> SimulatedAnnealing::optimize(){
        std::random_device rd;
        std::mt19937 mt(rd());
        std::uniform_int_distribution<> dis(0, this->searchSpace.size() - 1);

        auto selected = this->next(1000, dis(mt));

        Point<double> initial_solution = selected.first;
        Point<double> current_solution = initial_solution;
        Point<double> best_solution = initial_solution;
        
        double best_eval = this->objective(initial_solution);
        double current_eval = best_eval;
        double new_eval = best_eval;
        std::cout << initial_solution << " " << initial_solution.sum() << std::endl;
        std::cout << "Best eval: " << best_eval << std::endl;
        
        double t = this->T;
        size_t it;
        for(it = 0; it < max_iterations; it++){
            t *= this->alpha;
            //double t = this->T / (1 + this->alpha * it);
            double neighbors = 5000 * new_eval;
            
            if(neighbors < 1){
                break;
            }

            std::cout << "Neighbors: " << neighbors << " Objective distance: " << best_eval << std::endl;
            selected = this->next(neighbors, selected.second);
            Point<double> new_solution = selected.first;

            std::cout << "Current pos: " << selected.second << std::endl;
            
            new_eval = this->objective(new_solution);
            double diff = new_eval - current_eval;
            double metropolis = this->metropolis(diff, t);

            std::cout << std::endl;
            std::cout << "Iteration: " << it << std::endl;
            std::cout << "T: " << t << std::endl;
            std::cout << "Candidate solution: " << new_solution << std::endl;
            std::cout << "Candidate eval: " << new_eval << std::endl;
            std::cout << "Current solution: " << current_solution << std::endl;
            std::cout << "Current eval: " << current_eval << std::endl;
            std::cout << "Diff: " << diff << std::endl;
            std::cout << "Metropolis: " << metropolis << std::endl;
            std::cout << "Current best solution: " << current_solution << std::endl;
            std::cout << "Best eval: " << best_eval << std::endl;
            std::cout << "---------------------------------" << std::endl;
            std::cout << std::endl;

            if(new_eval < best_eval){
                best_eval = new_eval;
                best_solution = new_solution;
            }

            if(diff < 0 || mt() < metropolis){
                current_eval = new_eval;
                current_solution = new_solution;
            }

            if(best_eval == 0 || t <= this->minT){
                break;
            }
        }

        std::cout << "Best solution: " << best_solution << std::endl;
        std::cout << "Best eval: " << best_eval << std::endl;

        return best_solution;
    }

    double SimulatedAnnealing::metropolis(double diff, double t){
        return std::exp(-diff/t);
    }

    double SimulatedAnnealing::objective(mltk::Point<double> weights){
        thread_pool pool;
        Point<double> errors(10);
        auto k = this->k;
        std::vector<mltk::Data<double>> data(10);

        for(size_t i = 0; i < 10; i++){
            data[i] = this->samples.copy();
        }

        pool.parallelize_loop(0, 9, [&data, &errors, weights, k] (const int a, const int b) -> void {
            for(size_t i = a; i < b; i++){
                mltk::ensemble::kNNEnsembleW<double> knn_ensemb(data[i], k);
                knn_ensemb.setWeights(weights.X());
                
                double acc = mltk::validation::kfold(data[i], knn_ensemb, 10).accuracy/100;
                errors[i] += 1.0 - acc;
            }
        }, 10); 

        return errors.sum()/errors.size();
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

    void generateSearchSpaceAux(std::vector<double>& set, std::vector<double>& perm, std::vector<mltk::Point<double>>& space, int _k, double total_sum, int& N){
        auto sum = std::accumulate(perm.begin(), perm.end(), 0.0);

        if(sum > total_sum){
            return;
        }

        if(_k == 0){
            mltk::Point<double> sub_w(perm);
            if(sum == total_sum){
                space.push_back(sub_w);
                N++;
            }
            
            perm.clear();
            return;
        }
        for (int i = 0; i < set.size(); i++){
            std::vector<double> p = perm;
            p.push_back(set[i]);

            generateSearchSpaceAux(set, p, space, _k - 1, total_sum, N);
        }
    }

    std::vector<mltk::Point<double>> SimulatedAnnealing::generateSearchSpace(size_t size, double max_weight, double step){
        std::cout << size_t(1.0/step) << std::endl;
        std::vector<double> values = linspace(0, max_weight, size_t(1.0/step));
        std::vector<double> perm;
        std::vector<mltk::Point<double>> space;
        int N = 0;
        
        values.push_back(max_weight);
        values.push_back(max_weight/size);

        std::cout << "Generating search space..." << std::endl;
        std::cout << "Values: " << mltk::Point<double>(values) << std::endl;
        std::cout << std::endl;
        
        generateSearchSpaceAux(values, perm, space, size, max_weight, N);

        return space;
    }
};