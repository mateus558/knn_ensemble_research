//
// Created by mateu on 20/07/2021.
//

#ifndef KNN_RESEARCH_UTILS_H
#define KNN_RESEARCH_UTILS_H

#include <ufjfmltk/ufjfmltk.hpp>
#include <future>

mltk::Data<> load_dataset(const std::string& path, const std::string& prefix="", bool end = false);

void head(const mltk::Data<>& data, int n = 5);

std::mutex mutex;

mltk::Data<> load_dataset(const std::string& path, const std::string& prefix, bool end){
    mltk::Data<> data(prefix+path, end);

    mutex.lock();
    std::cout << "\nDataset name: " << data.name() << std::endl;
    std::cout << "size: " << data.size() << std::endl;
    std::cout << "dims: " << data.dim() << std::endl;
    std::cout << "classes: " << mltk::Point<int>(data.classes()) << std::endl;
    std::cout << "classes distribution: " << mltk::Point<size_t>(data.classesDistribution()) << std::endl;
    std::cout << std::endl;
    mutex.unlock();

    return data;
}

void head(const mltk::Data<>& data, int n){
    for(int i = 0; i < n; i++){
        std::cout << data(i) << std::endl;
    }
}

template <typename Callable>
void run(const std::vector<std::string>& datasets, bool *at_end, Callable experiment){
    int i = 0;
    std::vector<std::future<void> > futures(datasets.size());
    std::transform(datasets.begin(), datasets.end(), futures.begin(), [&](const std::string& path){
        return std::async(std::launch::async, experiment, path, at_end[i], i++);
    });
}

#endif //KNN_RESEARCH_UTILS_H
