#include "error_information_filter.h"

namespace slam_solver {

/* Specialized Template Class Declaration. */
template class ErrorInformationFilterDynamic<float>;
template class ErrorInformationFilterDynamic<double>;

/* Class Informaion Filer Definition. */
template <typename Scalar>
bool ErrorInformationFilterDynamic<Scalar>::PropagateInformationImpl() {
    dx_.setZero();
    const TMat<Scalar> F_t = F_.transpose();
    const TMat<Scalar> tmp = I_ + F_t * inverse_Q_ * F_;
    predict_I_ = inverse_Q_ - inverse_Q_ * F_ * tmp.ldlt().solve(F_t * inverse_Q_);
    return true;
}

template <typename Scalar>
bool ErrorInformationFilterDynamic<Scalar>::UpdateStateAndInformationImpl(const TMat<Scalar> &residual) {
    const TMat<Scalar> H_t = H_.transpose();

    // Update information.
    I_ = predict_I_ + H_t * inverse_R_ * H_;

    // Compute kalman gain.
    const TMat<Scalar> K_ = I_.ldlt().solve(H_t * inverse_R_);

    // Update error state.
    dx_ = K_ * residual;
    return true;
}

}  // namespace slam_solver
