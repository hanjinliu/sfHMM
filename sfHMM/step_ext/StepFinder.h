#ifndef STEP_H
#define STEP_H

#include <stdio.h>
#include <vector>
#include <cmath>
#include <limits>
#include <numeric>
#include <string>
#include <algorithm>

namespace stepc{
    class Base{
    public:
        std::vector<double> data;
        std::vector<double> fit;
        double penalty;
        int len;
        int n_step;
        std::vector<int> step_list;
        std::vector<int> len_list;
        std::vector<double> mu_list;
        std::vector<double> step_size_list;

        Base(std::vector<double>& rawdata, double p);
        void AddInfo();
    };

    class GaussStep: public Base{
    public:
        GaussStep(std::vector<double>& rawdata, double p);

        double OneStepFinding(int start, int end, int* pos);
        void AppendSteps();
        void MultiStepFinding();
    };
    
    class PoissonStep: public Base{
    public:
        PoissonStep(std::vector<double>& rawdata, double p);

        double OneStepFinding(int start, int end, int *pos);
        void AppendSteps(int start, int end);
        void MultiStepFinding();
    };
}

#endif