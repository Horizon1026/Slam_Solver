#include "slam_log_reporter.h"
#include "slam_operations.h"
#include "kalman_filter.h"
#include "error_kalman_filter.h"
#include "square_root_kalman_filter.h"
#include "information_filter.h"

#include <random>

using Scalar = float;
using namespace SLAM_SOLVER;

constexpr Scalar kMeasureNoiseSigma = 1.0;
constexpr Scalar kProcessNoiseSigma = 0.1;
constexpr Scalar kInitStateCovarianceSigma = 1e3f;
constexpr int32_t kNumberOfData = 1000;

void GenerateData(std::vector<Scalar> &truth_data,
                  std::vector<Scalar> &noised_data) {
    // Generate truth data.
    truth_data.resize(kNumberOfData);
    for (uint32_t i = 0; i < kNumberOfData; ++i) {
        truth_data[i] = 5.0f;
    }

    // Generate noised data.
    noised_data.resize(kNumberOfData);
    std::default_random_engine generator;
    std::normal_distribution<Scalar> noise(0., kMeasureNoiseSigma);
    for (uint32_t i = 0; i < noised_data.size(); ++i) {
        noised_data[i] = truth_data[i] + noise(generator);
    }
}

void PrintFilterResult(std::vector<Scalar> &truth_data,
                       std::vector<Scalar> &noised_data,
                       std::vector<Scalar> &filtered_data) {
    RETURN_IF(truth_data.size() != noised_data.size() || truth_data.size() != filtered_data.size());

    Scalar raw_noise = 0.0f;
    Scalar new_noise = 0.0f;
    for (uint32_t i = 0; i < noised_data.size(); ++i) {
        raw_noise += std::fabs(noised_data[i] - truth_data[i]);
        new_noise += std::fabs(filtered_data[i] - truth_data[i]);
    }
    raw_noise /= static_cast<Scalar>(noised_data.size());
    new_noise /= static_cast<Scalar>(filtered_data.size());
    ReportInfo("Noise of raw data and new data is " << raw_noise << " / " << new_noise);
}

template <typename FilterType>
void FilterNoisedDataInErrorState(const std::vector<Scalar> &noised_data,
                                  FilterType &filter,
                                  std::vector<Scalar> &filtered_data) {
    filtered_data = noised_data;
    for (uint32_t i = 1; i < noised_data.size(); ++i) {
        filter.PropagateNominalState();
        filter.PropagateCovariance();
        filter.UpdateStateAndCovariance(TVec1<Scalar>(noised_data[i] - filtered_data[i - 1]));
        filtered_data[i] = filter.dx()(0) + filtered_data[i - 1];
    }
}

template <typename FilterType>
void FilterNoisedDataInNominalState(const std::vector<Scalar> &noised_data,
                                    FilterType &filter,
                                    std::vector<Scalar> &filtered_data) {
    filtered_data = noised_data;
    filter.x() = TVec1<Scalar>(noised_data.front());
    for (uint32_t i = 1; i < noised_data.size(); ++i) {
        filter.PropagateNominalState();
        filter.PropagateCovariance();
        filter.UpdateStateAndCovariance(TVec1<Scalar>(noised_data[i]));
        filtered_data[i] = filter.x()(0);
    }
}

template <typename InverseFilterType>
void InverseFilterNoisedDataInNominalState(const std::vector<Scalar> &noised_data,
                                           InverseFilterType &filter,
                                           std::vector<Scalar> &filtered_data) {
    filtered_data = noised_data;
    filter.x() = TVec1<Scalar>(noised_data.front());
    for (uint32_t i = 1; i < noised_data.size(); ++i) {
        filter.PropagateNominalState();
        filter.PropagateInformation();
        filter.UpdateStateAndInformation(TVec1<Scalar>(noised_data[i]));
        filtered_data[i] = filter.x()(0);
    }
}

template <typename FilterType>
void InitializeKalmanFilter(FilterType &filter) {
    filter.options().kMethod = StateCovUpdateMethod::kFull;
    filter.P().setIdentity(1, 1);
    filter.P() *= kInitStateCovarianceSigma * kInitStateCovarianceSigma;
    filter.F().setIdentity(1, 1);
    filter.H().setIdentity(1, 1);
    filter.R() = TMat1<Scalar>(kMeasureNoiseSigma * kMeasureNoiseSigma);
    filter.Q() = TMat1<Scalar>(kProcessNoiseSigma * kProcessNoiseSigma);
}

template <typename FilterType>
void InitializeSquareRootKalmanFilter(FilterType &filter) {
    filter.options().kMethod = StateCovUpdateMethod::kFull;
    filter.S_t().setIdentity(1, 1);
    filter.S_t() *= kInitStateCovarianceSigma;
    filter.F().setIdentity(1, 1);
    filter.H().setIdentity(1, 1);
    filter.square_R_t() = TMat1<Scalar>(kMeasureNoiseSigma);
    filter.square_Q_t() = TMat1<Scalar>(kProcessNoiseSigma);
}

template <typename InverseFilterType>
void InitializeInformationFilter(InverseFilterType &filter) {
    filter.I().setIdentity(1, 1);
    filter.I() /= kInitStateCovarianceSigma * kInitStateCovarianceSigma;
    filter.F().setIdentity(1, 1);
    filter.H().setIdentity(1, 1);
    filter.inverse_R() = TMat1<Scalar>(1.0 / (kMeasureNoiseSigma * kMeasureNoiseSigma));
    filter.inverse_Q() = TMat1<Scalar>(1.0 / (kProcessNoiseSigma * kProcessNoiseSigma));
}

void TestKalmanFilterStatic(std::vector<Scalar> &truth_data,
                            std::vector<Scalar> &noised_data) {
    ReportInfo(YELLOW ">> Test kalman filter (static) in dimension 1." RESET_COLOR);
    KalmanFilterStatic<Scalar, 1, 1> filter;
    InitializeKalmanFilter(filter);
    std::vector<Scalar> filtered_data;
    FilterNoisedDataInNominalState(noised_data, filter, filtered_data);
    PrintFilterResult(truth_data, noised_data, filtered_data);
}

void TestErrorKalmanFilterStatic(std::vector<Scalar> &truth_data,
                                 std::vector<Scalar> &noised_data) {
    ReportInfo(YELLOW ">> Test error kalman filter (static) in dimension 1." RESET_COLOR);
    ErrorKalmanFilterStatic<Scalar, 1, 1> filter;
    InitializeKalmanFilter(filter);
    std::vector<Scalar> filtered_data;
    FilterNoisedDataInErrorState(noised_data, filter, filtered_data);
    PrintFilterResult(truth_data, noised_data, filtered_data);
}

void TestSquareRootKalmanFilterStatic(std::vector<Scalar> &truth_data,
                                      std::vector<Scalar> &noised_data) {
    ReportInfo(YELLOW ">> Test square root kalman filter (static) in dimension 1." RESET_COLOR);
    SquareRootKalmanFilterStatic<Scalar, 1, 1> filter;
    InitializeSquareRootKalmanFilter(filter);
    std::vector<Scalar> filtered_data;
    FilterNoisedDataInErrorState(noised_data, filter, filtered_data);
    PrintFilterResult(truth_data, noised_data, filtered_data);
}

void TestInformationFilterStatic(std::vector<Scalar> &truth_data,
                                 std::vector<Scalar> &noised_data) {
    ReportInfo(YELLOW ">> Test information filter (static) in dimension 1." RESET_COLOR);
    InformationFilterStatic<Scalar, 1, 1> filter;
    InitializeInformationFilter(filter);
    std::vector<Scalar> filtered_data;
    InverseFilterNoisedDataInNominalState(noised_data, filter, filtered_data);
    PrintFilterResult(truth_data, noised_data, filtered_data);
}

void TestKalmanFilterDynamic(std::vector<Scalar> &truth_data,
                             std::vector<Scalar> &noised_data) {
    ReportInfo(YELLOW ">> Test kalman filter (dynamic) in dimension 1." RESET_COLOR);
    KalmanFilterDynamic<Scalar> filter;
    InitializeKalmanFilter(filter);
    std::vector<Scalar> filtered_data;
    FilterNoisedDataInNominalState(noised_data, filter, filtered_data);
    PrintFilterResult(truth_data, noised_data, filtered_data);
}

void TestErrorKalmanFilterDynamic(std::vector<Scalar> &truth_data,
                                  std::vector<Scalar> &noised_data) {
    ReportInfo(YELLOW ">> Test error kalman filter (dynamic) in dimension 1." RESET_COLOR);
    ErrorKalmanFilterDynamic<Scalar> filter;
    InitializeKalmanFilter(filter);
    std::vector<Scalar> filtered_data;
    FilterNoisedDataInErrorState(noised_data, filter, filtered_data);
    PrintFilterResult(truth_data, noised_data, filtered_data);
}

void TestSquareRootKalmanFilterDynamic(std::vector<Scalar> &truth_data,
                                       std::vector<Scalar> &noised_data) {
    ReportInfo(YELLOW ">> Test square root kalman filter (dynamic) in dimension 1." RESET_COLOR);
    SquareRootKalmanFilterDynamic<Scalar> filter;
    InitializeSquareRootKalmanFilter(filter);
    std::vector<Scalar> filtered_data;
    FilterNoisedDataInErrorState(noised_data, filter, filtered_data);
    PrintFilterResult(truth_data, noised_data, filtered_data);
}

void TestInformationFilterDynamic(std::vector<Scalar> &truth_data,
                                  std::vector<Scalar> &noised_data) {
    ReportInfo(YELLOW ">> Test information filter (dynamic) in dimension 1." RESET_COLOR);
    InformationFilterDynamic<Scalar> filter;
    InitializeInformationFilter(filter);
    std::vector<Scalar> filtered_data;
    InverseFilterNoisedDataInNominalState(noised_data, filter, filtered_data);
    PrintFilterResult(truth_data, noised_data, filtered_data);
}

int main(int argc, char **argv) {
    ReportInfo(YELLOW ">> Test kalman filter solver." RESET_COLOR);

    std::vector<Scalar> truth_data;
    std::vector<Scalar> noised_data;
    GenerateData(truth_data, noised_data);

    TestKalmanFilterStatic(truth_data, noised_data);
    TestErrorKalmanFilterStatic(truth_data, noised_data);
    TestSquareRootKalmanFilterStatic(truth_data, noised_data);
    TestInformationFilterStatic(truth_data, noised_data);

    TestKalmanFilterDynamic(truth_data, noised_data);
    TestErrorKalmanFilterDynamic(truth_data, noised_data);
    TestSquareRootKalmanFilterDynamic(truth_data, noised_data);
    TestInformationFilterDynamic(truth_data, noised_data);

    return 0;
}
