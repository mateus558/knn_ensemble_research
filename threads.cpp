//
// Created by mateus on 19/07/2021.
//

#include <thread>
#include <future>
#include <execution>
#include <iostream>
#include <algorithm>

int main(){
    unsigned int n = std::thread::hardware_concurrency();
    std::cout << n << " concurrent threads are supported.\n";
}
