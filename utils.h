//
// Created by mateu on 20/07/2021.
//

#ifndef KNN_RESEARCH_UTILS_H
#define KNN_RESEARCH_UTILS_H

#include <ufjfmltk/ufjfmltk.hpp>
#include <future>

extern std::mutex mutex;
extern std::map<std::string, std::ofstream> log_files;

mltk::Data<> load_dataset(const std::string& path, const std::string& prefix="", bool end = false, bool print_info=true);

void head(const mltk::Data<>& data, int n = 5);

template <typename Callable>
inline void run(const std::vector<std::string>& datasets, bool *at_end, Callable experiment){
    int i = -1;
    std::vector<std::future<void> > futures(datasets.size());
    std::transform(datasets.begin(), datasets.end(), futures.begin(), [&](const std::string& path){
        i++;
        return std::async(std::launch::async, experiment, path, at_end[i], i);
    });
}

#endif //KNN_RESEARCH_UTILS_H
