#include "error_information_filter.h"

namespace SLAM_SOLVER {

/* Specialized Template Class Declaration. */
template class ErrorInformationFilterDynamic<float>;
template class ErrorInformationFilterDynamic<double>;

/* Class Informaion Filer Definition. */
template <typename Scalar>
bool ErrorInformationFilterDynamic<Scalar>::PropagateNominalStateImpl(const TVec<Scalar> &parameters) {
    // For error state filter, nominal state propagation should not only use F_.
    return true;
}

template <typename Scalar>
bool ErrorInformationFilterDynamic<Scalar>::PropagateInformationImpl() {
    const TMat<Scalar> F_t = F_.transpose();
    predict_I_ = inverse_Q_ - inverse_Q_ * F_ * (I_ + F_t * inverse_Q_ * F_).inverse() * F_t * inverse_Q_;
    return true;
}

template <typename Scalar>
bool ErrorInformationFilterDynamic<Scalar>::UpdateStateAndInformationImpl(const TMat<Scalar> &residual) {
    const TMat<Scalar> H_t = H_.transpose();

    // Update information.
    I_ = predict_I_ + H_t * inverse_R_ * H_;

    // Compute kalman gain.
    const TMat<Scalar> K_ = I_.inverse() * H_t * inverse_R_;

    // Update error state.
    dx_ = K_ * residual;
    return true;
}

}
