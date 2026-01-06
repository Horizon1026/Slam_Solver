#ifndef _KALMAN_FILTER_SOLVER_H_
#define _KALMAN_FILTER_SOLVER_H_

#include "basic_type.h"
#include "filter.h"

namespace slam_solver {

struct KalmanFilterOptions {
    StateCovUpdateMethod kMethod = StateCovUpdateMethod::kSimple;
};

/**
 * @brief Kalman Filter (KF)
 * 
 * References:
 * - "A New Approach to Linear Filtering and Prediction Problems", Kalman, 1960.
 * - "Optimal State Estimation: Kalman, H Infinity, and Nonlinear Approaches", Dan Simon.
 * 
 * Algorithm Flow:
 * 1. Predict:
 *    - x_pre = F * x
 *    - P_pre = F * P * F^T + Q
 * 2. Update:
 *    - S = H * P_pre * H^T + R
 *    - K = P_pre * H^T * S^-1
 *    - x = x_pre + K * (z - H * x_pre)
 *    - P = (I - K * H) * P_pre (Simple)
 *    - P = (I - K * H) * P_pre * (I - K * H)^T + K * R * K^T (Joseph form, Full)
 * 
 * Variables:
 * - x: State vector
 * - P: State covariance matrix
 * - F: State transition matrix
 * - H: Measurement matrix
 * - Q: Process noise covariance matrix
 * - R: Measurement noise covariance matrix
 * - z: Measurement vector
 * - K: Kalman gain
 * - S: Innovation covariance
 */
template <typename Scalar>
class KalmanFilterDynamic: public Filter<Scalar, KalmanFilterDynamic<Scalar>> {

public:
    KalmanFilterDynamic(): Filter<Scalar, KalmanFilterDynamic<Scalar>>() {}
    virtual ~KalmanFilterDynamic() = default;

    bool PropagateCovarianceImpl();
    bool UpdateStateAndCovarianceImpl(const TMat<Scalar> &observation = TMat<Scalar>::Zero(1, 1));

    // Reference for member variables.
    KalmanFilterOptions &options() { return options_; }
    TVec<Scalar> &x() { return x_; }
    TMat<Scalar> &P() { return P_; }
    TVec<Scalar> &predict_x() { return predict_x_; }
    TMat<Scalar> &predict_P() { return predict_P_; }
    TMat<Scalar> &F() { return F_; }
    TMat<Scalar> &H() { return H_; }
    TMat<Scalar> &Q() { return Q_; }
    TMat<Scalar> &R() { return R_; }

    // Const reference for member variables.
    const KalmanFilterOptions &options() const { return options_; }
    const TVec<Scalar> &x() const { return x_; }
    const TMat<Scalar> &P() const { return P_; }
    const TVec<Scalar> &predict_x() const { return predict_x_; }
    const TMat<Scalar> &predict_P() const { return predict_P_; }
    const TMat<Scalar> &F() const { return F_; }
    const TMat<Scalar> &H() const { return H_; }
    const TMat<Scalar> &Q() const { return Q_; }
    const TMat<Scalar> &R() const { return R_; }

private:
    KalmanFilterOptions options_;

    TVec<Scalar> x_ = TVec<Scalar>::Zero(1, 1);
    TMat<Scalar> P_ = TMat<Scalar>::Zero(1, 1);

    TVec<Scalar> predict_x_ = TMat<Scalar>::Zero(1, 1);
    TMat<Scalar> predict_P_ = TMat<Scalar>::Zero(1, 1);
    TMat<Scalar> predict_S_ = TMat<Scalar>::Zero(1, 1);

    // Process function F and measurement function H.
    TMat<Scalar> F_ = TMat<Scalar>::Identity(1, 1);
    TMat<Scalar> H_ = TMat<Scalar>::Identity(1, 1);

    // Process noise Q and measurement noise R.
    TMat<Scalar> Q_ = TMat<Scalar>::Zero(1, 1);
    TMat<Scalar> R_ = TMat<Scalar>::Zero(1, 1);
};

/**
 * @brief Static Dimensional Kalman Filter (KF)
 * @tparam StateSize Dimension of the state vector
 * @tparam ObserveSize Dimension of the measurement vector
 * 
 * Algorithm and variables same as KalmanFilterDynamic.
 */
template <typename Scalar, int32_t StateSize, int32_t ObserveSize>
class KalmanFilterStatic: public Filter<Scalar, KalmanFilterStatic<Scalar, StateSize, ObserveSize>> {

    static_assert(StateSize > 0 && ObserveSize > 0, "Size of state and observe must be larger than 0.");

public:
    KalmanFilterStatic(): Filter<Scalar, KalmanFilterStatic<Scalar, StateSize, ObserveSize>>() {}
    virtual ~KalmanFilterStatic() = default;

    bool PropagateCovarianceImpl();
    bool UpdateStateAndCovarianceImpl(const TMat<Scalar> &observation = TMat<Scalar>::Zero(1, 1));

    // Reference for member variables.
    KalmanFilterOptions &options() { return options_; }
    TVec<Scalar, StateSize> &x() { return x_; }
    TMat<Scalar, StateSize, StateSize> &P() { return P_; }
    TVec<Scalar, StateSize> &predict_x() { return predict_x_; }
    TMat<Scalar, StateSize, StateSize> &predict_P() { return predict_P_; }
    TMat<Scalar, StateSize, StateSize> &F() { return F_; }
    TMat<Scalar, ObserveSize, StateSize> &H() { return H_; }
    TMat<Scalar, StateSize, StateSize> &Q() { return Q_; }
    TMat<Scalar, ObserveSize, ObserveSize> &R() { return R_; }

    // Const reference for member variables.
    const KalmanFilterOptions &options() const { return options_; }
    const TVec<Scalar, StateSize> &x() const { return x_; }
    const TMat<Scalar, StateSize, StateSize> &P() const { return P_; }
    const TVec<Scalar, StateSize> &predict_x() const { return predict_x_; }
    const TMat<Scalar, StateSize, StateSize> &predict_P() const { return predict_P_; }
    const TMat<Scalar, StateSize, StateSize> &F() const { return F_; }
    const TMat<Scalar, ObserveSize, StateSize> &H() const { return H_; }
    const TMat<Scalar, StateSize, StateSize> &Q() const { return Q_; }
    const TMat<Scalar, ObserveSize, ObserveSize> &R() const { return R_; }

private:
    KalmanFilterOptions options_;

    TVec<Scalar, StateSize> x_ = TVec<Scalar, StateSize>::Zero();
    TMat<Scalar, StateSize, StateSize> P_ = TMat<Scalar, StateSize, StateSize>::Zero();

    TVec<Scalar, StateSize> predict_x_ = TVec<Scalar, StateSize>::Zero();
    TMat<Scalar, StateSize, StateSize> predict_P_ = TMat<Scalar, StateSize, StateSize>::Zero();
    TMat<Scalar, ObserveSize, ObserveSize> predict_S_ = TMat<Scalar, ObserveSize, ObserveSize>::Zero();

    // Process function F and measurement function H.
    TMat<Scalar, StateSize, StateSize> F_ = TMat<Scalar, StateSize, StateSize>::Identity();
    TMat<Scalar, ObserveSize, StateSize> H_ = TMat<Scalar, ObserveSize, StateSize>::Zero();

    // Process noise Q and measurement noise R.
    TMat<Scalar, StateSize, StateSize> Q_ = TMat<Scalar, StateSize, StateSize>::Zero();
    TMat<Scalar, ObserveSize, ObserveSize> R_ = TMat<Scalar, ObserveSize, ObserveSize>::Zero();
};

/* Class Basic Kalman Filter Definition. */
template <typename Scalar, int32_t StateSize, int32_t ObserveSize>
bool KalmanFilterStatic<Scalar, StateSize, ObserveSize>::PropagateCovarianceImpl() {
    predict_x_ = F_ * x_;
    predict_P_ = F_ * P_ * F_.transpose() + Q_;
    return true;
}

template <typename Scalar, int32_t StateSize, int32_t ObserveSize>
bool KalmanFilterStatic<Scalar, StateSize, ObserveSize>::UpdateStateAndCovarianceImpl(const TMat<Scalar> &observation) {
    const TMat<Scalar, StateSize, ObserveSize> H_t = H_.transpose();

    // Compute Kalman gain.
    predict_S_ = H_ * predict_P_ * H_t + R_;
    const TMat<Scalar, StateSize, ObserveSize> K_ = predict_P_ * H_t * predict_S_.ldlt().solve(TMat<Scalar, ObserveSize, ObserveSize>::Identity());

    // Update new state.
    const TVec<Scalar, ObserveSize> v_ = observation - H_ * predict_x_;
    x_ = predict_x_ + K_ * v_;

    // Update covariance of new state.
    TMat<Scalar, StateSize, StateSize> I_KH = TMat<Scalar, StateSize, StateSize>::Identity() - K_ * H_;
    switch (options_.kMethod) {
        default:
        case StateCovUpdateMethod::kSimple:
            P_ = I_KH * predict_P_;
            break;
        case StateCovUpdateMethod::kFull:
            P_ = I_KH * predict_P_ * I_KH.transpose() + K_ * R_ * K_.transpose();
            break;
    }

    // Maintenance of symmetry.
    P_ = (P_ + P_.transpose()) * static_cast<Scalar>(0.5);

    return true;
}

}  // namespace slam_solver

#endif  // end of _KALMAN_FILTER_SOLVER_H_
