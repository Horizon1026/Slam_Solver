#include "information_filter.h"

namespace slam_solver {

/* Specialized Template Class Declaration. */
template class InformationFilterDynamic<float>;
template class InformationFilterDynamic<double>;

/* Class Informaion Filer Definition. */
template <typename Scalar>
bool InformationFilterDynamic<Scalar>::PropagateInformationImpl() {
    predict_x_ = F_ * x_;
    const TMat<Scalar> F_t = F_.transpose();
    const TMat<Scalar> tmp = I_ + F_t * inverse_Q_ * F_;
    predict_I_ = inverse_Q_ - inverse_Q_ * F_ * tmp.ldlt().solve(F_t * inverse_Q_);
    return true;
}

template <typename Scalar>
bool InformationFilterDynamic<Scalar>::UpdateStateAndInformationImpl(const TMat<Scalar> &observation) {
    const TMat<Scalar> H_t = H_.transpose();

    // Update information.
    I_ = predict_I_ + H_t * inverse_R_ * H_;

    // Compute kalman gain.
    const TMat<Scalar> K_ = I_.ldlt().solve(H_t * inverse_R_);

    // Update new state.
    const TVec<Scalar> v_ = observation - H_ * predict_x_;
    x_ = predict_x_ + K_ * v_;
    return true;
}

}  // namespace slam_solver
