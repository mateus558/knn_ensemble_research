//
// Created by mateu on 30/07/2021.
//

#include "utils.h"

std::mutex mutex;
std::map<std::string, std::ofstream> log_files;
thread_pool pool(20);
synced_stream synced_cout;

mltk::Data<> load_dataset(const std::string& path, const std::string& prefix, bool end, bool print_info){
    std::cout << prefix+path << std::endl;
    mltk::Data<> data(prefix+path, end);

    if(print_info) {
        std::stringstream point;
        point << mltk::Point<int>(data.classes());
        synced_cout.println("\nPath: " + prefix + path);
        synced_cout.println("Dataset name: " + data.name());
        synced_cout.println("size: " + std::to_string(data.size()));
        synced_cout.println("dims: " + std::to_string(data.dim()));
        synced_cout.println("classes: " + point.str());
        point.clear();
        point.str("");
        point << mltk::Point<size_t>(data.classesDistribution());
        synced_cout.println("classes distribution: " + point.str());
        synced_cout.println();
        synced_cout.println("------------------------------------------------------");
        //synced_cout.println("dataset head:");
        //head(data);
    }
    return data;
}

void head(const mltk::Data<>& data, int n){
    std::stringstream ss;
    for(int i = 0; i < n; i++){
        ss << data(i);
        synced_cout.println(ss.str());
        ss.clear();
        ss.str("");
    }
}
