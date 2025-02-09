#include "information_filter.h"

namespace SLAM_SOLVER {

/* Specialized Template Class Declaration. */
template class InformationFilterDynamic<float>;
template class InformationFilterDynamic<double>;

/* Class Informaion Filer Definition. */
template <typename Scalar>
bool InformationFilterDynamic<Scalar>::PropagateNominalStateImpl(const TVec<Scalar> &parameters) {
    predict_x_ = F_ * x_;
    return true;
}

template <typename Scalar>
bool InformationFilterDynamic<Scalar>::PropagateInformationImpl() {
    const TMat<Scalar> F_t = F_.transpose();
    predict_I_ = inverse_Q_ - inverse_Q_ * F_ * (I_ + F_t * inverse_Q_ * F_).inverse() * F_t * inverse_Q_;
    return true;
}

template <typename Scalar>
bool InformationFilterDynamic<Scalar>::UpdateStateAndInformationImpl(const TMat<Scalar> &observation) {
    const TMat<Scalar> H_t = H_.transpose();

    // Update information.
    I_ = predict_I_ + H_t * inverse_R_ * H_;

    // Compute kalman gain.
    const TMat<Scalar> K_ = I_.inverse() * H_t * inverse_R_;

    // Update new state.
    const TVec<Scalar> v_ = observation - H_ * predict_x_;
    x_ = predict_x_ + K_ * v_;
    return true;
}

}
