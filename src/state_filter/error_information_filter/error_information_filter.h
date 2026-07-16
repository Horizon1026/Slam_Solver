#ifndef _ERROR_INFORMATION_FILTER_SOLVER_H_
#define _ERROR_INFORMATION_FILTER_SOLVER_H_

#include "basic_type.h"
#include "inverse_filter.h"

namespace slam_solver {

/**
 * @brief Error State Information Filter (EIF)
 *
 * Algorithm Flow:
 * 1. Predict:
 *    - dx = 0 (reset error state)
 *    - I_pre = Q^-1 - Q^-1 * F * (I + F^T * Q^-1 * F)^-1 * F^T * Q^-1
 * 2. Update:
 *    - I = I_pre + H^T * R^-1 * H
 *    - K = I^-1 * H^T * R^-1
 *    - dx = K * residual
 *
 * Variables:
 * - dx: Error state vector
 * - I: Error state information matrix (P^-1)
 * - F: Error state transition matrix
 * - H: Measurement Jacobian matrix
 * - inverse_Q: Process noise information matrix (Q^-1)
 * - inverse_R: Measurement noise information matrix (R^-1)
 */
template <typename Scalar>
class ErrorInformationFilterDynamic: public InverseFilter<Scalar, ErrorInformationFilterDynamic<Scalar>> {

public:
    ErrorInformationFilterDynamic(): InverseFilter<Scalar, ErrorInformationFilterDynamic<Scalar>>() {}
    virtual ~ErrorInformationFilterDynamic() = default;

    bool PropagateInformationImpl();
    bool UpdateStateAndInformationImpl(const TMat<Scalar> &residual = TMat<Scalar>::Zero(1, 1));

    // Reference for member variables.
    TVec<Scalar> &dx() { return dx_; }
    TMat<Scalar> &I() { return I_; }
    TMat<Scalar> &predict_I() { return predict_I_; }
    TMat<Scalar> &F() { return F_; }
    TMat<Scalar> &H() { return H_; }
    TMat<Scalar> &inverse_Q() { return inverse_Q_; }
    TMat<Scalar> &inverse_R() { return inverse_R_; }
    TMat<Scalar> &null_space() { return null_space_; }

    // Const reference for member variables.
    const TVec<Scalar> &dx() const { return dx_; }
    const TMat<Scalar> &I() const { return I_; }
    const TMat<Scalar> &predict_I() const { return predict_I_; }
    const TMat<Scalar> &F() const { return F_; }
    const TMat<Scalar> &H() const { return H_; }
    const TMat<Scalar> &inverse_Q() const { return inverse_Q_; }
    const TMat<Scalar> &inverse_R() const { return inverse_R_; }
    const TMat<Scalar> &null_space() const { return null_space_; }

private:
    TVec<Scalar> dx_ = TVec<Scalar>::Zero(1, 1);
    TMat<Scalar> I_ = TMat<Scalar>::Zero(1, 1);
    TMat<Scalar> predict_I_ = TMat<Scalar>::Zero(1, 1);

    // Process function F and measurement function H.
    TMat<Scalar> F_ = TMat<Scalar>::Identity(1, 1);
    TMat<Scalar> H_ = TMat<Scalar>::Identity(1, 1);

    // Process noise Q and measurement noise R.
    TMat<Scalar> inverse_Q_ = TMat<Scalar>::Zero(1, 1);
    TMat<Scalar> inverse_R_ = TMat<Scalar>::Zero(1, 1);

    // Null space matrix for projecting Kalman gain.
    // When set (cols > 0), the Kalman gain is projected as
    // K_proj = (I - N * (N^T*N)^{-1} * N^T) * K,
    // so that states in the column space of N are unaffected by the observation.
    TMat<Scalar> null_space_ = TMat<Scalar>::Zero(0, 0);
};

/**
 * @brief Static Dimensional Error Information Filter (EIF)
 * @tparam StateSize Dimension of the error state vector
 * @tparam ObserveSize Dimension of the measurement vector
 *
 * Algorithm and variables same as ErrorInformationFilterDynamic.
 */
template <typename Scalar, int32_t StateSize, int32_t ObserveSize>
class ErrorInformationFilterStatic: public InverseFilter<Scalar, ErrorInformationFilterStatic<Scalar, StateSize, ObserveSize>> {

    static_assert(StateSize > 0 && ObserveSize > 0, "Size of state and observe must be larger than 0.");

public:
    ErrorInformationFilterStatic(): InverseFilter<Scalar, ErrorInformationFilterStatic<Scalar, StateSize, ObserveSize>>() {}
    virtual ~ErrorInformationFilterStatic() = default;

    bool PropagateInformationImpl();
    bool UpdateStateAndInformationImpl(const TMat<Scalar> &observation = TMat<Scalar>::Zero(1, 1));

    // Reference for member variables.
    TVec<Scalar, StateSize> &dx() { return dx_; }
    TMat<Scalar, StateSize, StateSize> &I() { return I_; }
    TMat<Scalar, StateSize, StateSize> &predict_I() { return predict_I_; }
    TMat<Scalar, StateSize, StateSize> &F() { return F_; }
    TMat<Scalar, ObserveSize, StateSize> &H() { return H_; }
    TMat<Scalar, StateSize, StateSize> &inverse_Q() { return inverse_Q_; }
    TMat<Scalar, ObserveSize, ObserveSize> &inverse_R() { return inverse_R_; }
    TMat<Scalar, StateSize, Eigen::Dynamic> &null_space() { return null_space_; }

    // Const reference for member variables.
    const TVec<Scalar, StateSize> &dx() const { return dx_; }
    const TMat<Scalar, StateSize, StateSize> &I() const { return I_; }
    const TMat<Scalar, StateSize, StateSize> &predict_I() const { return predict_I_; }
    const TMat<Scalar, StateSize, StateSize> &F() const { return F_; }
    const TMat<Scalar, ObserveSize, StateSize> &H() const { return H_; }
    const TMat<Scalar, StateSize, StateSize> &inverse_Q() const { return inverse_Q_; }
    const TMat<Scalar, ObserveSize, ObserveSize> &inverse_R() const { return inverse_R_; }
    const TMat<Scalar, StateSize, Eigen::Dynamic> &null_space() const { return null_space_; }

private:
    TVec<Scalar, StateSize> dx_ = TVec<Scalar, StateSize>::Zero();
    TMat<Scalar, StateSize, StateSize> I_ = TMat<Scalar, StateSize, StateSize>::Zero();
    TMat<Scalar, StateSize, StateSize> predict_I_ = TMat<Scalar, StateSize, StateSize>::Zero();

    // Process function F and measurement function H.
    TMat<Scalar, StateSize, StateSize> F_ = TMat<Scalar, StateSize, StateSize>::Identity();
    TMat<Scalar, ObserveSize, StateSize> H_ = TMat<Scalar, ObserveSize, StateSize>::Zero();

    // Process noise Q and measurement noise R.
    TMat<Scalar, StateSize, StateSize> inverse_Q_ = TMat<Scalar, StateSize, StateSize>::Zero();
    TMat<Scalar, ObserveSize, ObserveSize> inverse_R_ = TMat<Scalar, ObserveSize, ObserveSize>::Zero();

    // Null space matrix for projecting Kalman gain.
    // When set (cols > 0), the Kalman gain is projected as
    // K_proj = (I - N * (N^T*N)^{-1} * N^T) * K,
    // so that states in the column space of N are unaffected by the observation.
    TMat<Scalar, StateSize, Eigen::Dynamic> null_space_;
};

/* Class Basic Information Filter Definition. */
template <typename Scalar, int32_t StateSize, int32_t ObserveSize>
bool ErrorInformationFilterStatic<Scalar, StateSize, ObserveSize>::PropagateInformationImpl() {
    dx_.setZero();
    const TMat<Scalar, StateSize, StateSize> F_t = F_.transpose();
    const TMat<Scalar, StateSize, StateSize> tmp = I_ + F_t * inverse_Q_ * F_;
    predict_I_ = inverse_Q_ - inverse_Q_ * F_ * tmp.ldlt().solve(F_t * inverse_Q_);
    return true;
}

template <typename Scalar, int32_t StateSize, int32_t ObserveSize>
bool ErrorInformationFilterStatic<Scalar, StateSize, ObserveSize>::UpdateStateAndInformationImpl(const TMat<Scalar> &residual) {
    const TMat<Scalar, StateSize, ObserveSize> H_t = H_.transpose();

    // Update information.
    I_ = predict_I_ + H_t * inverse_R_ * H_;

    // Compute kalman gain.
    TMat<Scalar, StateSize, ObserveSize> K_ = I_.ldlt().solve(H_t * inverse_R_);

    // Project Kalman gain using null space (if set).
    // K_proj = (I - N * (N^T*N)^{-1} * N^T) * K
    // States in the column space of null_space_ will not be affected by the observation.
    if (null_space_.cols() > 0) {
        const TMat<Scalar> N = null_space_;
        const TMat<Scalar> NtN = N.transpose() * N;
        K_ -= N * NtN.ldlt().solve(N.transpose() * K_);

        // Recompute information matrix to be consistent with the null-space-projected gain.
        // The standard info update I = predict_I + H^T * R^{-1} * H adds information in all
        // observable directions, but the null space states should retain their prior uncertainty.
        // Using the Joseph-form covariance update with K_proj:
        //   P_new = (I - K_proj*H) * P_pred * (I - K_proj*H)^T + K_proj * R * K_proj^T
        // and then converting back: I_new = P_new^{-1}.
        const TMat<Scalar, StateSize, StateSize> P_pred = predict_I_.ldlt().solve(TMat<Scalar, StateSize, StateSize>::Identity());
        const TMat<Scalar, StateSize, StateSize> I_KH = TMat<Scalar, StateSize, StateSize>::Identity() - K_ * H_;
        const TMat<Scalar, ObserveSize, ObserveSize> R_mat = inverse_R_.ldlt().solve(TMat<Scalar, ObserveSize, ObserveSize>::Identity());
        const TMat<Scalar, StateSize, StateSize> P_new = I_KH * P_pred * I_KH.transpose() + K_ * R_mat * K_.transpose();
        I_ = P_new.ldlt().solve(TMat<Scalar, StateSize, StateSize>::Identity());

        // Maintenance of symmetry.
        I_ = (I_ + I_.transpose()) * static_cast<Scalar>(0.5);
    }

    // Update error state.
    dx_ = K_ * residual;
    return true;
}

}  // namespace slam_solver

#endif  // end of _ERROR_INFORMATION_FILTER_SOLVER_H_
