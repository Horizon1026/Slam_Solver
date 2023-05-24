#include "log_report.h"
#include "kalman_filter.h"
#include "error_kalman_filter.h"

#include <random>

using Scalar = float;
using namespace SLAM_SOLVER;

constexpr Scalar kNoiseSigma = 1.0f;
constexpr int32_t kNumberOfData = 1000;

void GenerateData(std::vector<Scalar> &truth_data,
                  std::vector<Scalar> &noised_data) {
    // Generate truth data.
    truth_data.resize(kNumberOfData);
    for (uint32_t i = 0; i < kNumberOfData; ++i) {
        truth_data[i] = std::sin(i * 3.1415926f * 0.001f) * 5.0f;
    }

    // Generate noised data.
    noised_data.resize(kNumberOfData);
    std::default_random_engine generator;
    std::normal_distribution<Scalar> noise(0., kNoiseSigma);
    for (uint32_t i = 0; i < noised_data.size(); ++i) {
        noised_data[i] = truth_data[i] + noise(generator);
    }
}

void PrintFilterResult(std::vector<Scalar> &truth_data,
                       std::vector<Scalar> &noised_data,
                       std::vector<Scalar> &filtered_data) {
    Scalar raw_noise = 0.0f;
    Scalar new_noise = 0.0f;
    for (uint32_t i = 0; i < noised_data.size(); ++i) {
        raw_noise += std::fabs(noised_data[i] - truth_data[i]);
        new_noise += std::fabs(filtered_data[i] - truth_data[i]);
        // ReportInfo("[noised | filterd] data is [" << noised_data[i] << " | " << filtered_data[i] << "]");
    }
    raw_noise /= static_cast<Scalar>(noised_data.size());
    new_noise /= static_cast<Scalar>(filtered_data.size());
    ReportInfo("Noise of raw data and new data is " << raw_noise << " / " << new_noise);
}

void TestKalmanFilter(std::vector<Scalar> &truth_data,
                      std::vector<Scalar> &noised_data) {
    ReportInfo(YELLOW ">> Test kalman filter in dimension 1." RESET_COLOR);


    // Construct filter for this data.
    KalmanFilter<Scalar, 1, 1> filter;
    filter.options().kMethod = StateCovUpdateMethod::kFull;
    filter.F().setIdentity();
    filter.R() = TMat<Scalar, 1, 1>(kNoiseSigma);
    filter.Q() = TMat<Scalar, 1, 1>(0.01f);
    filter.H().setIdentity();

    // Filter noised data.
    std::vector<Scalar> filtered_data = noised_data;
    filter.x() = TVec<Scalar, 1>(noised_data.front());
    for (uint32_t i = 1; i < noised_data.size(); ++i) {
        filter.PropagateNominalState();
        filter.PropagateCovariance();
        filter.UpdateStateAndCovariance(TVec<Scalar, 1>(noised_data[i]));
        filtered_data[i] = filter.x()(0);
    }

    // Print result.
    PrintFilterResult(truth_data, noised_data, filtered_data);
}

void TestErrorKalmanFilter(std::vector<Scalar> &truth_data,
                           std::vector<Scalar> &noised_data) {
    ReportInfo(YELLOW ">> Test error kalman filter in dimension 1." RESET_COLOR);


    // Construct filter for this data.
    ErrorKalmanFilter<Scalar, 1, 1> filter;
    filter.options().kMethod = StateCovUpdateMethod::kFull;
    filter.F().setIdentity();
    filter.R() = TMat<Scalar, 1, 1>(kNoiseSigma);
    filter.Q() = TMat<Scalar, 1, 1>(0.01f);
    filter.H().setIdentity();

    // Filter noised data.
    std::vector<Scalar> filtered_data = noised_data;
    for (uint32_t i = 1; i < noised_data.size(); ++i) {
        filter.PropagateNominalState();
        filter.PropagateCovariance();
        filter.UpdateStateAndCovariance(TVec<Scalar, 1>(noised_data[i] - filtered_data[i - 1]));
        filtered_data[i] = filter.dx()(0) + filtered_data[i - 1];
    }

    // Print result.
    PrintFilterResult(truth_data, noised_data, filtered_data);
}

int main(int argc, char **argv) {
    ReportInfo(YELLOW ">> Test kalman filter solver." RESET_COLOR);

    std::vector<Scalar> truth_data;
    std::vector<Scalar> noised_data;
    GenerateData(truth_data, noised_data);

    TestKalmanFilter(truth_data, noised_data);
    TestErrorKalmanFilter(truth_data, noised_data);

    return 0;
}
