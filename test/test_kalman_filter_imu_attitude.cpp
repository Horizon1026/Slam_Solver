#include "log_report.h"
#include "math_kinematics.h"
#include "imu_state.h"
#include "error_kalman_filter.h"
#include "square_root_kalman_filter.h"

#include <fstream>

using namespace SLAM_UTILITY;
using namespace SENSOR_MODEL;
using namespace SLAM_SOLVER;

namespace {
    constexpr float kGyroNoiseSigma = 0.01f;
    constexpr float kGyroRandomWalkSigma = 0.001f;
}

bool LoadImuMeasurements(const std::string &imu_file,
                         std::vector<ImuMeasurement> &measurements,
                         std::vector<Vec3> &position,
                         std::vector<Quat> &rotation) {
    ReportInfo(">> Load imu data from " << imu_file);

    measurements.clear();
    position.clear();
    rotation.clear();

    std::ifstream fsIMU;
    fsIMU.open(imu_file.c_str());
    if (!fsIMU.is_open()) {
        std::cout << "   failed." << std::endl;
        return false;
    }

    std::string oneLine;
    double time_stamp;
    TVec3<double> acc, gyr, pos;
    TQuat<double> q;
    uint32_t cnt = 0;
    while (std::getline(fsIMU, oneLine) && !oneLine.empty()) {
        std::istringstream imuData(oneLine);
        imuData >> time_stamp >> q.w() >> q.x() >> q.y() >> q.z() >> pos.x() >> pos.y() >> pos.z()
			>> gyr.x() >> gyr.y() >> gyr.z() >> acc.x() >> acc.y() >> acc.z();

        ImuMeasurement meas;
        meas.accel = acc.cast<float>();
        meas.gyro = gyr.cast<float>();
        meas.time_stamp = time_stamp;
        measurements.emplace_back(meas);
        position.emplace_back(pos.cast<float>());
        rotation.emplace_back(q.cast<float>());

        ++cnt;
    }

    ReportInfo(">> " GREEN << cnt << RESET_COLOR " imu raw data loaded.");
    return true;
}

void TestErrorKalmanFilter(const std::vector<ImuMeasurement> &meas,
                           std::vector<Quat> &est_q,
                           std::vector<Vec3> &est_bw) {
	ReportInfo(YELLOW ">> Test error state kalman filter." RESET_COLOR);

	// Set initial state.
    est_q.resize(meas.size());
    est_q.front().setIdentity();
    est_bw.resize(meas.size());
    est_bw.front().setZero();

	// Initialize filter.
    ErrorKalmanFilterStatic<float, 6, 3> filter;
    filter.F().setIdentity();
    filter.H().setZero();
    filter.R().setIdentity();
    filter.Q().setIdentity();

    for (uint32_t i = 1; i < meas.size(); ++i) {
		const Vec3 gyro = 0.5f * (meas[i - 1].gyro + meas[i].gyro) - est_bw[i - 1];
        const float dt = meas[i].time_stamp - meas[i - 1].time_stamp;

        // Propagate nominal state.
        est_q[i] = est_q[i - 1] * Utility::DeltaQ(gyro * dt);
        est_q[i].normalize();
        est_bw[i] = est_bw[i - 1];
        filter.PropagateNominalState();

        // Propagate covariance.
        filter.F().block<3, 3>(0, 0) = Mat3::Identity() - Utility::SkewSymmetricMatrix(gyro - est_bw[i - 1]) * dt;
        filter.F().block<3, 3>(0, 3) = -dt * Mat3::Identity();
        filter.Q().block<3, 3>(0, 0) = Mat3::Identity() * kGyroNoiseSigma * kGyroNoiseSigma * dt * dt;
        filter.Q().block<3, 3>(3, 3) = Mat3::Identity() * kGyroRandomWalkSigma * kGyroRandomWalkSigma * dt * dt;
        filter.PropagateCovariance();

        // Update state and covariance with observations.
        const Vec3 obv = meas[i].accel / meas[i].accel.norm();
        const Vec3 pred = est_q[i].matrix().transpose().col(2);
        const Vec3 residual = Utility::SkewSymmetricMatrix(pred) * obv;
        filter.H().block<3, 3>(0, 0) = Utility::SkewSymmetricMatrix(obv) * Utility::SkewSymmetricMatrix(pred);
        const float weight = std::fabs(meas[i].accel.norm() - 9.81f);
        filter.R() = Mat3::Identity() * (weight * weight * weight + 0.001f);
		filter.UpdateStateAndCovariance(residual);

        est_q[i] = est_q[i] * Utility::DeltaQ(filter.dx().head<3>());
        est_q[i].normalize();
        est_bw[i] = est_bw[i] + filter.dx().tail<3>();
    }

}

void TestSquareRootKalmanFilter(const std::vector<ImuMeasurement> &meas,
                                std::vector<Quat> &est_q,
                                std::vector<Vec3> &est_bw) {
	ReportInfo(YELLOW ">> Test error state square root kalman filter." RESET_COLOR);

	// Set initial state.
    est_q.resize(meas.size());
    est_q.front().setIdentity();
    est_bw.resize(meas.size());
    est_bw.front().setZero();

	// Initialize filter.
    SquareRootKalmanFilterStatic<float, 6, 3> filter;
    filter.F().setIdentity();
    filter.H().setZero();
    filter.square_R_t().setIdentity();
    filter.square_Q_t().setIdentity();

    for (uint32_t i = 1; i < meas.size(); ++i) {
		const Vec3 gyro = 0.5f * (meas[i - 1].gyro + meas[i].gyro) - est_bw[i - 1];
        const float dt = meas[i].time_stamp - meas[i - 1].time_stamp;

        // Propagate nominal state.
        est_q[i] = est_q[i - 1] * Utility::DeltaQ(gyro * dt);
        est_q[i].normalize();
        est_bw[i] = est_bw[i - 1];
        filter.PropagateNominalState();

        // Propagate covariance.
        filter.F().block<3, 3>(0, 0) = Mat3::Identity() - Utility::SkewSymmetricMatrix(gyro - est_bw[i - 1]) * dt;
        filter.F().block<3, 3>(0, 3) = -dt * Mat3::Identity();
        filter.square_Q_t().block<3, 3>(0, 0) = Mat3::Identity() * kGyroNoiseSigma * dt;
        filter.square_Q_t().block<3, 3>(3, 3) = Mat3::Identity() * kGyroRandomWalkSigma * dt;
        filter.PropagateCovariance();

        // Update state and covariance with observations.
        const Vec3 obv = meas[i].accel / meas[i].accel.norm();
        const Vec3 pred = est_q[i].matrix().transpose().col(2);
        const Vec3 residual = Utility::SkewSymmetricMatrix(pred) * obv;
        filter.H().block<3, 3>(0, 0) = Utility::SkewSymmetricMatrix(obv) * Utility::SkewSymmetricMatrix(pred);
        const float weight = std::fabs(meas[i].accel.norm() - 9.81f);
        filter.square_R_t() = Mat3::Identity() * std::sqrt(weight * weight * weight + 0.001f);
		filter.UpdateStateAndCovariance(residual);

        est_q[i] = est_q[i] * Utility::DeltaQ(filter.dx().head<3>());
        est_q[i].normalize();
        est_bw[i] = est_bw[i] + filter.dx().tail<3>();
    }
}

void ComputeEstimationResidual(const std::vector<Quat> &truth,
							   const std::vector<Quat> &estimate) {
    if (truth.size() != estimate.size()) {
        return;
    }

    std::array<Vec3, 3> mean_min_max = {};
    const Quat res_q = truth.front().inverse() * estimate.front();
    const Vec3 res_euler = Utility::QuaternionToEuler(res_q);
	mean_min_max[0] = Vec3::Ones() * res_euler.x();
    mean_min_max[1] = Vec3::Ones() * res_euler.y();
    mean_min_max[2] = Vec3::Ones() * res_euler.z();

    for (uint32_t i = 0; i < truth.size(); ++i) {
        const Quat q = truth[i].inverse() * estimate[i];
        const Vec3 euler = Utility::QuaternionToEuler(q);
		mean_min_max[0](0) += euler.x();
        mean_min_max[1](0) += euler.y();
        mean_min_max[2](0) += euler.z();
        mean_min_max[0](1) = std::min(mean_min_max[0](1), euler.x());
        mean_min_max[1](1) = std::min(mean_min_max[1](1), euler.y());
        mean_min_max[2](1) = std::min(mean_min_max[2](1), euler.z());
        mean_min_max[0](2) = std::max(mean_min_max[0](2), euler.x());
        mean_min_max[1](2) = std::max(mean_min_max[1](2), euler.y());
        mean_min_max[2](2) = std::max(mean_min_max[2](2), euler.z());
    }

	for (uint32_t i = 0; i < 3; ++i) {
    	mean_min_max[i](0) /= static_cast<float>(truth.size());
    }

    ReportInfo("The mean/min/max residual of pitch is " << LogVec(mean_min_max[0]));
    ReportInfo("The mean/min/max residual of roll is " << LogVec(mean_min_max[1]));
    ReportInfo("The mean/min/max residual of yaw is " << LogVec(mean_min_max[2]));
}

int main(int argc, char **argv) {
    std::string imu_file;
    if (argc == 2) {
        imu_file = argv[1];
    }

    ReportInfo(YELLOW ">> Test kalman filter on pure imu attitude estimation." RESET_COLOR);
    LogFixPercision(5);

    std::vector<ImuMeasurement> measurements;
    std::vector<Vec3> position;
    std::vector<Quat> rotation;
    if (LoadImuMeasurements(imu_file, measurements, position, rotation) == false) {
        return 0;
    }

    std::vector<Quat> estimation_q;
    std::vector<Vec3> estimation_bw;

    TestErrorKalmanFilter(measurements, estimation_q, estimation_bw);
    ComputeEstimationResidual(rotation, estimation_q);
    estimation_bw.clear();
    estimation_q.clear();

    TestSquareRootKalmanFilter(measurements, estimation_q, estimation_bw);
    ComputeEstimationResidual(rotation, estimation_q);
    estimation_bw.clear();
    estimation_q.clear();

    return 0;
}
