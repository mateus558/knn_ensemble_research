//
// Created by mateu on 20/07/2021.
//

#ifndef KNN_RESEARCH_UTILS_H
#define KNN_RESEARCH_UTILS_H

#include <ufjfmltk/ufjfmltk.hpp>
#include "thread_pool.hpp"

extern std::mutex mutex;
extern std::map<std::string, std::ofstream> log_files;
extern thread_pool pool;

mltk::Data<> load_dataset(const std::string& path, const std::string& prefix="", bool end = false, bool print_info=true);

void head(const mltk::Data<>& data, int n = 5);

template <typename Callable>
inline void run(const std::vector<std::string>& datasets, bool *at_end, Callable experiment){
    int i = -1;

    std::for_each(datasets.begin(), datasets.end(), [&](const std::string& path){
        i++;
        pool.push_task(experiment, path, at_end[i], i);
    });
}

#endif //KNN_RESEARCH_UTILS_H
