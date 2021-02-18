#include "StepFinder.h"

/*
Step finding algorithm by Kalafut et al. (2008) is modified here.
BIC is not appropriate because fitting result would largely depend on data size.
I introduced transition probability p and steps are detected based on likelihood maximization
likelihood = Gauss(data-gauss)*p^k*(1-p)*(N-k)
where k is the number of steps and N is the number of data.
*/

const double INF = std::numeric_limits<double>::infinity();

double VectorSqrsum(int start, int end, const std::vector<double>& vec){
    // sum(vec[start:end]^2)
    if(start >= end){
        return 0.0;
    }
    double sum1, sum2;
    auto itr = vec.begin();
    sum1 = std::accumulate(itr + start, itr + end, 0.0);
    sum2 = std::inner_product(itr + start, itr + end, itr + start, 0.0);
    return sum2 - sum1 * sum1 / static_cast<double>(end - start);
}

double VectorMean(int start, int end, const std::vector<double>& vec){
    if(start >= end){
        return 0.0;
    }
    auto itr = vec.begin();
    return std::accumulate(itr + start, itr + end, 0.0) / static_cast<double>(end - start);
}

double VectorXlogX(int start, int end, const std::vector<double>& vec){
    // sum*log(mean)
    double mean = VectorMean(start, end, vec);
    if(mean == 0){
        return 0;
    }
    else{
        return static_cast<double>(end - start) * mean * std::log(mean);
    }
}

// XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
// XXX Base class
// XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
stepc::Base::Base(std::vector<double>& rawdata, double p) {
    data = rawdata;
    len = data.size();
    if (0.0 < p && p < 1.0){
        penalty = std::log(p/(1-p));
    }
    else{
        penalty = -0.5 * std::log(len);
    }
    step_list = {0, len};
    n_step = 0;
    fit = std::vector<double>(len, 0.0);
    len_list = {};
    mu_list = {};
    step_size_list = {};
}


void stepc::Base::AddInfo(){
    double mean;
    int start, end;
    n_step = step_list.size() - 1;
    mu_list = std::vector<double>(n_step, 0);
    len_list = std::vector<int>(n_step, 0);
    step_size_list = std::vector<double>(n_step - 1, 0);

    for (int i = 0; i < n_step; i++){
        start = step_list[i];
        end = step_list[i + 1];
        mean = VectorMean(start, end, data);
        mu_list[i] = mean;
        len_list[i] = end - start;
        for (int j = start; j < end; j++){
            fit[j] = mean;
        }
    }
    for (int i = 0; i < n_step - 1; i++){
        step_size_list[i] = mu_list[i + 1] - mu_list[i];
    }
}

// XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
// XXX class GaussStep
// XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

stepc::GaussStep::GaussStep(std::vector<double>& rawdata, double p) : Base(rawdata, p){};

double stepc::GaussStep::OneStepFinding(int start, int end, int* pos){
    double oldSqrsum;
    double NewSqrsum;
    double bestSqrsum = INF;

    oldSqrsum = VectorSqrsum(start, end, data);
    for (int p = start + 1; p < end; p++){
        NewSqrsum = VectorSqrsum(start, p, data) + VectorSqrsum(p, end, data);
        if (NewSqrsum < bestSqrsum){
            bestSqrsum = NewSqrsum;
            *pos = p;
        }
    }

    return bestSqrsum - oldSqrsum;
}

void stepc::GaussStep::AppendSteps(){
    int i = 0;
    bool repeat_flag = true;
    int last_updated = 0;
    double dL;
    int StepPos;
    double sqrsum = VectorSqrsum(0, len, data);
    double dsqrsum;
    int start;
    int end;
    std::vector<int> BestStepPos = {0};
    std::vector<double> Bestdsqrsum = {0};

    while(repeat_flag){
        start = step_list[i];
        end = step_list[i + 1];
        if (end - start > 2){
            if(BestStepPos[i] > 0){
                StepPos = BestStepPos[i];
                dsqrsum = Bestdsqrsum[i];
            }
            else{
                dsqrsum = OneStepFinding(start, end, &StepPos);
                BestStepPos[i] = StepPos;
                Bestdsqrsum[i] = dsqrsum;
            }
            
            dL = penalty - len / 2 * (std::log(1 + dsqrsum / sqrsum));
            if (dL > 0) {
                step_list.insert(step_list.begin() + i + 1, StepPos);
                BestStepPos[i] = 0;
                BestStepPos.insert(BestStepPos.begin() + i + 1, 0);
                Bestdsqrsum.insert(Bestdsqrsum.begin() + i + 1, 0);
                sqrsum = sqrsum + dsqrsum;
                last_updated = i + 1;
            }
            else if(i + 1 == last_updated || step_list.size() == 2){
                repeat_flag = false;
            }
        }
        else if(i + 1 == last_updated){
            repeat_flag = false;
        }

        i = i + 1;
        // back to the beginning
        if(i >= step_list.size() - 1){
            i = 0;
        }
    }
}


void stepc::GaussStep::MultiStepFinding(){
    AppendSteps();
    AddInfo();
}

// XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
// XXX class PoissonStep
// XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

stepc::PoissonStep::PoissonStep(std::vector<double>& rawdata, double p) : Base(rawdata, p){};

double stepc::PoissonStep::OneStepFinding(int start, int end, int* pos){
    double oldXlogX;
    double newXlogX;
    double bestXlogX = -1 * INF;

    oldXlogX = VectorXlogX(start, end, data);
    for (int p = start + 1; p < end; p++){
        newXlogX = VectorXlogX(start, p, data) + VectorXlogX(p, end, data);
        if (newXlogX > bestXlogX){
            bestXlogX = newXlogX;
            *pos = p;
        }
    }

    return bestXlogX - oldXlogX;
}

void stepc::PoissonStep::AppendSteps(int start, int end){
    if(end - start < 3){
        return;
    }
    double dL;
    int StepPos;
    double dXlogX;

    dXlogX = OneStepFinding(start, end, &StepPos);
    dL = penalty + dXlogX;
    if(dL > 0){
        step_list.push_back(StepPos);
        AppendSteps(start, StepPos);
        AppendSteps(StepPos, end);
    }

}


void stepc::PoissonStep::MultiStepFinding(){
    AppendSteps(0, len);
    std::sort(step_list.begin(), step_list.end());
    AddInfo();
}