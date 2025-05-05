#ifndef _ERROR_KALMAN_FILTER_SOLVER_H_
#define _ERROR_KALMAN_FILTER_SOLVER_H_

#include "basic_type.h"
#include "filter.h"

namespace SLAM_SOLVER {

struct ErrorKalmanFilterOptions {
    StateCovUpdateMethod kMethod = StateCovUpdateMethod::kSimple;
};

/* Class Error State Kalman Filter Declaration. */
template <typename Scalar>
class ErrorKalmanFilterDynamic : public Filter<Scalar, ErrorKalmanFilterDynamic<Scalar>> {

public:
    ErrorKalmanFilterDynamic() : Filter<Scalar, ErrorKalmanFilterDynamic<Scalar>>() {}
    virtual ~ErrorKalmanFilterDynamic() = default;

    bool PropagateNominalStateImpl(const TVec<Scalar> &parameters = TMat<Scalar>::Zero(1, 1));
    bool PropagateCovarianceImpl();
    bool UpdateStateAndCovarianceImpl(const TMat<Scalar> &observation = TMat<Scalar>::Zero(1, 1));

    // Reference for member variables.
    ErrorKalmanFilterOptions &options() { return options_; }
    TVec<Scalar> &dx() { return dx_; }
    TMat<Scalar> &P() { return P_; }
    TMat<Scalar> &predict_P() { return predict_P_; }
    TMat<Scalar> &F() { return F_; }
    TMat<Scalar> &H() { return H_; }
    TMat<Scalar> &Q() { return Q_; }
    TMat<Scalar> &R() { return R_; }

    // Const reference for member variables.
    const ErrorKalmanFilterOptions &options() const { return options_; }
    const TVec<Scalar> &dx() const { return dx_; }
    const TMat<Scalar> &P() const { return P_; }
    const TMat<Scalar> &predict_P() const { return predict_P_; }
    const TMat<Scalar> &F() const { return F_; }
    const TMat<Scalar> &H() const { return H_; }
    const TMat<Scalar> &Q() const { return Q_; }
    const TMat<Scalar> &R() const { return R_; }

private:
    ErrorKalmanFilterOptions options_;

    TVec<Scalar> dx_ = TVec<Scalar>::Zero(1, 1);
    TMat<Scalar> P_ = TMat<Scalar>::Zero(1, 1);

    TMat<Scalar> predict_P_ = TMat<Scalar>::Zero(1, 1);
    TMat<Scalar> predict_S_ = TMat<Scalar>::Zero(1, 1);

    // Process function F and measurement function H.
    TMat<Scalar> F_ = TMat<Scalar>::Identity(1, 1);
    TMat<Scalar> H_ = TMat<Scalar>::Identity(1, 1);

    // Process noise Q and measurement noise R.
    TMat<Scalar> Q_ = TMat<Scalar>::Zero(1, 1);
    TMat<Scalar> R_ = TMat<Scalar>::Zero(1, 1);

};

/* Class Error State Kalman Filter Declaration. */
template <typename Scalar, int32_t StateSize, int32_t ObserveSize>
class ErrorKalmanFilterStatic : public Filter<Scalar, ErrorKalmanFilterStatic<Scalar, StateSize, ObserveSize>> {

static_assert(StateSize > 0 && ObserveSize > 0, "Size of state and observe must be larger than 0.");

public:
    ErrorKalmanFilterStatic() : Filter<Scalar, ErrorKalmanFilterStatic<Scalar, StateSize, ObserveSize>>() {}
    virtual ~ErrorKalmanFilterStatic() = default;

    bool PropagateNominalStateImpl(const TVec<Scalar> &parameters = TMat<Scalar>::Zero(1, 1));
    bool PropagateCovarianceImpl();
    bool UpdateStateAndCovarianceImpl(const TMat<Scalar> &observation = TMat<Scalar>::Zero(1, 1));

    // Reference for member variables.
    ErrorKalmanFilterOptions &options() { return options_; }
    TVec<Scalar, StateSize> &dx() { return dx_; }
    TMat<Scalar, StateSize, StateSize> &P() { return P_; }
    TMat<Scalar, StateSize, StateSize> &predict_P() { return predict_P_; }
    TMat<Scalar, StateSize, StateSize> &F() { return F_; }
    TMat<Scalar, ObserveSize, StateSize> &H() { return H_; }
    TMat<Scalar, StateSize, StateSize> &Q() { return Q_; }
    TMat<Scalar, ObserveSize, ObserveSize> &R() { return R_; }

    // Const reference for member variables.
    const ErrorKalmanFilterOptions &options() const { return options_; }
    const TVec<Scalar, StateSize> &dx() const { return dx_; }
    const TMat<Scalar, StateSize, StateSize> &P() const { return P_; }
    const TMat<Scalar, StateSize, StateSize> &predict_P() const { return predict_P_; }
    const TMat<Scalar, StateSize, StateSize> &F() const { return F_; }
    const TMat<Scalar, ObserveSize, StateSize> &H() const { return H_; }
    const TMat<Scalar, StateSize, StateSize> &Q() const { return Q_; }
    const TMat<Scalar, ObserveSize, ObserveSize> &R() const { return R_; }

private:
    ErrorKalmanFilterOptions options_;

    TVec<Scalar, StateSize> dx_ = TVec<Scalar, StateSize>::Zero();
    TMat<Scalar, StateSize, StateSize> P_ = TMat<Scalar, StateSize, StateSize>::Zero();

    TMat<Scalar, StateSize, StateSize> predict_P_ = TMat<Scalar, StateSize, StateSize>::Zero();
    TMat<Scalar, ObserveSize, ObserveSize> predict_S_ = TMat<Scalar, ObserveSize, ObserveSize>::Zero();

    // Process function F and measurement function H.
    TMat<Scalar, StateSize, StateSize> F_ = TMat<Scalar, StateSize, StateSize>::Identity();
    TMat<Scalar, ObserveSize, StateSize> H_ = TMat<Scalar, ObserveSize, StateSize>::Identity();

    // Process noise Q and measurement noise R.
    TMat<Scalar, StateSize, StateSize> Q_ = TMat<Scalar, StateSize, StateSize>::Zero();
    TMat<Scalar, ObserveSize, ObserveSize> R_ = TMat<Scalar, ObserveSize, ObserveSize>::Zero();

};

/* Class Error State Kalman Filter Definition. */
template <typename Scalar, int32_t StateSize, int32_t ObserveSize>
bool ErrorKalmanFilterStatic<Scalar, StateSize, ObserveSize>::PropagateNominalStateImpl(const TVec<Scalar> &parameters) {
    // For error state filter, nominal state propagation should not only use F_.
    return true;
}

template <typename Scalar, int32_t StateSize, int32_t ObserveSize>
bool ErrorKalmanFilterStatic<Scalar, StateSize, ObserveSize>::PropagateCovarianceImpl() {
    predict_P_ = F_ * P_ * F_.transpose() + Q_;
    return true;
}

template <typename Scalar, int32_t StateSize, int32_t ObserveSize>
bool ErrorKalmanFilterStatic<Scalar, StateSize, ObserveSize>::UpdateStateAndCovarianceImpl(const TMat<Scalar> &residual) {
    // Compute Kalman gain.
    const TMat<Scalar, StateSize, ObserveSize> H_t = H_.transpose();
    predict_S_ = H_ * predict_P_ * H_t + R_;
    const TMat<Scalar, StateSize, ObserveSize> K_ = predict_P_ * H_t * predict_S_.inverse();

    // Update error state.
    dx_ = K_ * residual;

    // Update covariance of new state.
    TMat<Scalar, StateSize, StateSize> I_KH = -K_ * H_;
    I_KH.diagonal().array() += static_cast<Scalar>(1);
    switch (options_.kMethod) {
        default:
        case StateCovUpdateMethod::kSimple:
            P_ = I_KH * predict_P_;
            break;
        case StateCovUpdateMethod::kFull:
            P_ = I_KH * predict_P_ * I_KH.transpose() + K_ * R_ * K_.transpose();
            break;
    }

    return true;
}

}

#endif // end of _ERROR_KALMAN_FILTER_SOLVER_H_
