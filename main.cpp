#include <iostream>
#include "Experiment/Experiment.hpp"


int main(int argc, char** argv) {
    Experiment app{"App description"};

    app.add_parameters();

    CLI11_PARSE(app, argc, argv);

    app.run();

    return 0;
}