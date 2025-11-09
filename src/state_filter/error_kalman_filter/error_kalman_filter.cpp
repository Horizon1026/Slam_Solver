#include "error_kalman_filter.h"

namespace slam_solver {

/* Specialized Template Class Declaration. */
template class ErrorKalmanFilterDynamic<float>;
template class ErrorKalmanFilterDynamic<double>;

/* Class Basic Kalman Filter Definition. */
template <typename Scalar>
bool ErrorKalmanFilterDynamic<Scalar>::PropagateCovarianceImpl() {
    predict_P_ = F_ * P_ * F_.transpose() + Q_;
    return true;
}

template <typename Scalar>
bool ErrorKalmanFilterDynamic<Scalar>::UpdateStateAndCovarianceImpl(const TMat<Scalar> &residual) {
    // Compute Kalman gain.
    const TMat<Scalar> H_t = H_.transpose();
    predict_S_ = H_ * predict_P_ * H_t + R_;
    const TMat<Scalar> K_ = predict_P_ * H_t * predict_S_.inverse();

    // Update error state.
    dx_ = K_ * residual;

    // Update covariance of new state.
    TMat<Scalar> I_KH = -K_ * H_;
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

}  // namespace slam_solver
