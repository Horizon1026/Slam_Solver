#include "kalman_filter.h"

namespace SLAM_SOLVER {

/* Specialized Template Class Declaration. */
template class KalmanFilterDynamic<float>;
template class KalmanFilterDynamic<double>;

/* Class Basic Kalman Filter Definition. */
template <typename Scalar>
bool KalmanFilterDynamic<Scalar>::PropagateNominalStateImpl(const TVec<Scalar> &parameters) {
    predict_x_ = F_ * x_;
    return true;
}

template <typename Scalar>
bool KalmanFilterDynamic<Scalar>::PropagateCovarianceImpl() {
    predict_P_ = F_ * P_ * F_.transpose() + Q_;
    return true;
}

template <typename Scalar>
bool KalmanFilterDynamic<Scalar>::UpdateStateAndCovarianceImpl(const TMat<Scalar> &observation) {
    const TMat<Scalar> H_t = H_.transpose();

    // Compute Kalman gain.
    predict_S_ = H_ * predict_P_ * H_t + R_;
    const TMat<Scalar> K_ = predict_P_ * H_t * predict_S_.inverse();

    // Compute new information.
    const TVec<Scalar> v_ = observation - H_ * predict_x_;

    // Update new state.
    x_ = predict_x_ + K_ * v_;

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

}
