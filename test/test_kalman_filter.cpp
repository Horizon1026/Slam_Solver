#include "log_api.h"
#include "kalman_filter.h"

#include <random>

using Scalar = float;
using namespace SLAM_SOLVER;

void TestKalmanFilter() {
    LogInfo(YELLOW ">> Test kalman filter in dimension 1." RESET_COLOR);
    constexpr Scalar kNoiseSigma = 1.0f;
    constexpr Scalar kTruthValue = 8.0f;

    // Generate noised data.
    std::vector<Scalar> noised_data;
    noised_data.clear();
    noised_data.resize(1000, kTruthValue);
    std::default_random_engine generator;
    std::normal_distribution<Scalar> noise(0., kNoiseSigma);
    for (uint32_t i = 0; i < noised_data.size(); ++i) {
        noised_data[i] += noise(generator);
    }

    // Construct filter for this data.
    KalmanFilter<Scalar, 1, 1> filter;
    filter.F().setIdentity();
    filter.R() = TMat<Scalar, 1, 1>(kNoiseSigma);
    filter.Q() = TMat<Scalar, 1, 1>(0.01f);
    filter.H().setIdentity();

    // Filter noised data.
    std::vector<Scalar> filtered_data = noised_data;
    filter.x() = TVec<Scalar, 1>(noised_data.front());
    for (uint32_t i = 0; i < noised_data.size(); ++i) {
        filter.Propagate();
        filter.Update(TVec<Scalar, 1>(noised_data[i]));
        filtered_data[i] = filter.x()(0);
    }

    // Print result.
    Scalar raw_noise = 0.0f;
    Scalar new_noise = 0.0f;
    for (uint32_t i = 0; i < noised_data.size(); ++i) {
        raw_noise += std::fabs(noised_data[i] - kTruthValue);
        new_noise += std::fabs(filtered_data[i] - kTruthValue);
        // LogInfo("[noised | filterd] data is [" << noised_data[i] << " | " << filtered_data[i] << "]");
    }
    raw_noise /= static_cast<Scalar>(noised_data.size());
    new_noise /= static_cast<Scalar>(filtered_data.size());
    LogInfo("Noise of raw data and new data is " << raw_noise << " / " << new_noise);
}

int main(int argc, char **argv) {
    LogInfo(YELLOW ">> Test kalman filter solver." RESET_COLOR);

    TestKalmanFilter();

    return 0;
}
