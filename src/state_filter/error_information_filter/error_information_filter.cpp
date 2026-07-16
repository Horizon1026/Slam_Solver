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
    TMat<Scalar> K_ = I_.ldlt().solve(H_t * inverse_R_);

    // Project Kalman gain using null space (if set).
    // K_proj = (I - N * (N^T*N)^{-1} * N^T) * K
    // States in the column space of null_space_ will not be affected by the observation.
    if (null_space_.cols() > 0) {
        const TMat<Scalar> N = null_space_;
        const TMat<Scalar> NtN = N.transpose() * N;
        K_ -= N * NtN.ldlt().solve(N.transpose() * K_);

        // Recompute information matrix to be consistent with the null-space-projected gain.
        const TMat<Scalar> P_pred = predict_I_.ldlt().solve(TMat<Scalar>::Identity(predict_I_.rows(), predict_I_.cols()));
        const TMat<Scalar> I_KH = TMat<Scalar>::Identity(I_.rows(), I_.cols()) - K_ * H_;
        const TMat<Scalar> R_mat = inverse_R_.ldlt().solve(TMat<Scalar>::Identity(inverse_R_.rows(), inverse_R_.cols()));
        const TMat<Scalar> P_new = I_KH * P_pred * I_KH.transpose() + K_ * R_mat * K_.transpose();
        I_ = P_new.ldlt().solve(TMat<Scalar>::Identity(P_new.rows(), P_new.cols()));

        // Maintenance of symmetry.
        I_ = (I_ + I_.transpose()) * static_cast<Scalar>(0.5);
    }

    // Update error state.
    dx_ = K_ * residual;
    return true;
}

}  // namespace slam_solver
