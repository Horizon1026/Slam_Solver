#include "kalman_filter.h"

namespace slam_solver {

/* Specialized Template Class Declaration. */
template class KalmanFilterDynamic<float>;
template class KalmanFilterDynamic<double>;

/* Class Basic Kalman Filter Definition. */
template <typename Scalar>
bool KalmanFilterDynamic<Scalar>::PropagateCovarianceImpl() {
    predict_x_ = F_ * x_;
    predict_P_ = F_ * P_ * F_.transpose() + Q_;
    return true;
}

template <typename Scalar>
bool KalmanFilterDynamic<Scalar>::UpdateStateAndCovarianceImpl(const TMat<Scalar> &observation) {
    const TMat<Scalar> H_t = H_.transpose();

    // Compute Kalman gain.
    predict_S_ = H_ * predict_P_ * H_t + R_;
    const TMat<Scalar> K_ = predict_P_ * H_t * predict_S_.ldlt().solve(TMat<Scalar>::Identity(predict_S_.rows(), predict_S_.cols()));

    // Update new state.
    const TVec<Scalar> v_ = observation - H_ * predict_x_;
    x_ = predict_x_ + K_ * v_;

    // Update covariance of new state.
    TMat<Scalar> I_KH = TMat<Scalar>::Identity(x_.rows(), x_.rows()) - K_ * H_;
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
