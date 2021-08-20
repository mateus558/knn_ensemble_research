//
// Created by mateu on 30/07/2021.
//

#include "utils.h"

std::mutex mutex;
std::map<std::string, std::ofstream> log_files;
thread_pool pool(10);

mltk::Data<> load_dataset(const std::string& path, const std::string& prefix, bool end, bool print_info){
    mltk::Data<> data(prefix+path, end);

    if(print_info) {
        mutex.lock();
        std::cout << "\nPath: " << prefix + path << std::endl;
        std::cout << "Dataset name: " << data.name() << std::endl;
        std::cout << "size: " << data.size() << std::endl;
        std::cout << "dims: " << data.dim() << std::endl;
        std::cout << "classes: " << mltk::Point<int>(data.classes()) << std::endl;
        std::cout << "classes distribution: " << mltk::Point<size_t>(data.classesDistribution()) << std::endl;
        std::cout << std::endl;
        mutex.unlock();
    }
    return data;
}

void head(const mltk::Data<>& data, int n){
    for(int i = 0; i < n; i++){
        std::cout << data(i) << std::endl;
    }
}
