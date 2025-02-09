#ifndef _ERROR_INFORMATION_FILTER_SOLVER_H_
#define _ERROR_INFORMATION_FILTER_SOLVER_H_

#include "basic_type.h"
#include "inverse_filter.h"

namespace SLAM_SOLVER {

/* Class Error Information Filter Declaration. */
template <typename Scalar>
class ErrorInformationFilterDynamic : public InverseFilter<Scalar, ErrorInformationFilterDynamic<Scalar>> {

public:
    ErrorInformationFilterDynamic() : InverseFilter<Scalar, ErrorInformationFilterDynamic<Scalar>>() {}
    virtual ~ErrorInformationFilterDynamic() = default;

    bool PropagateNominalStateImpl(const TVec<Scalar> &parameters = TMat<Scalar>::Zero(1, 1));
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

    // Const reference for member variables.
    const TVec<Scalar> &dx() const { return dx_; }
    const TMat<Scalar> &I() const { return I_; }
    const TMat<Scalar> &predict_I() const { return predict_I_; }
    const TMat<Scalar> &F() const { return F_; }
    const TMat<Scalar> &H() const { return H_; }
    const TMat<Scalar> &inverse_Q() const { return inverse_Q_; }
    const TMat<Scalar> &inverse_R() const { return inverse_R_; }

private:
    TVec<Scalar> dx_ = TMat<Scalar>::Zero(1, 1);
    TMat<Scalar> I_ = TMat<Scalar>::Zero(1, 1);
    TMat<Scalar> predict_I_ = TMat<Scalar>::Zero(1, 1);

    // Process function F and measurement function H.
    TMat<Scalar> F_ = TMat<Scalar>::Identity(1, 1);
    TMat<Scalar> H_ = TMat<Scalar>::Identity(1, 1);

    // Process noise Q and measurement noise R.
    TMat<Scalar> inverse_Q_ = TMat<Scalar>::Zero(1, 1);
    TMat<Scalar> inverse_R_ = TMat<Scalar>::Zero(1, 1);

};

/* Class Basic Information Filer Declaration. */
template <typename Scalar, int32_t StateSize, int32_t ObserveSize>
class ErrorInformationFilterStatic : public InverseFilter<Scalar, ErrorInformationFilterStatic<Scalar, StateSize, ObserveSize>> {

public:
    ErrorInformationFilterStatic() : InverseFilter<Scalar, ErrorInformationFilterStatic<Scalar, StateSize, ObserveSize>>() {}
    virtual ~ErrorInformationFilterStatic() = default;

    bool PropagateNominalStateImpl(const TVec<Scalar> &parameters = TMat<Scalar>::Zero(1, 1));
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

    // Const reference for member variables.
    const TVec<Scalar, StateSize> &dx() const { return dx_; }
    const TMat<Scalar, StateSize, StateSize> &I() const { return I_; }
    const TMat<Scalar, StateSize, StateSize> &predict_I() const { return predict_I_; }
    const TMat<Scalar, StateSize, StateSize> &F() const { return F_; }
    const TMat<Scalar, ObserveSize, StateSize> &H() const { return H_; }
    const TMat<Scalar, StateSize, StateSize> &inverse_Q() const { return inverse_Q_; }
    const TMat<Scalar, ObserveSize, ObserveSize> &inverse_R() const { return inverse_R_; }

private:
    TVec<Scalar, StateSize> dx_ = TVec<Scalar, StateSize>::Zero();
    TMat<Scalar, StateSize, StateSize> I_ = TMat<Scalar, StateSize, StateSize>::Zero();
    TMat<Scalar, StateSize, StateSize> predict_I_ = TMat<Scalar, StateSize, StateSize>::Zero();

    // Process function F and measurement function H.
    TMat<Scalar, StateSize, StateSize> F_ = TMat<Scalar, StateSize, StateSize>::Identity();
    TMat<Scalar, ObserveSize, StateSize> H_ = TMat<Scalar, ObserveSize, StateSize>::Identity();

    // Process noise Q and measurement noise R.
    TMat<Scalar, StateSize, StateSize> inverse_Q_ = TMat<Scalar, StateSize, StateSize>::Zero();
    TMat<Scalar, ObserveSize, ObserveSize> inverse_R_ = TMat<Scalar, ObserveSize, ObserveSize>::Zero();
};

/* Class Basic Information Filter Definition. */
template <typename Scalar, int32_t StateSize, int32_t ObserveSize>
bool ErrorInformationFilterStatic<Scalar, StateSize, ObserveSize>::PropagateNominalStateImpl(const TVec<Scalar> &parameters) {
    // For error state filter, nominal state propagation should not only use F_.
    return true;
}

template <typename Scalar, int32_t StateSize, int32_t ObserveSize>
bool ErrorInformationFilterStatic<Scalar, StateSize, ObserveSize>::PropagateInformationImpl() {
    const TMat<Scalar, StateSize, StateSize> F_t = F_.transpose();
    predict_I_ = inverse_Q_ - inverse_Q_ * F_ * (I_ + F_t * inverse_Q_ * F_).inverse() * F_t * inverse_Q_;
    return true;
}

template <typename Scalar, int32_t StateSize, int32_t ObserveSize>
bool ErrorInformationFilterStatic<Scalar, StateSize, ObserveSize>::UpdateStateAndInformationImpl(const TMat<Scalar> &residual) {
    const TMat<Scalar, StateSize, ObserveSize> H_t = H_.transpose();

    // Update information.
    I_ = predict_I_ + H_t * inverse_R_ * H_;

    // Compute kalman gain.
    const TMat<Scalar, StateSize, ObserveSize> K_ = I_.inverse() * H_t * inverse_R_;

    // Update error state.
    dx_ = K_ * residual;
    return true;
}

}

#endif // end of _ERROR_INFORMATION_FILTER_SOLVER_H_
