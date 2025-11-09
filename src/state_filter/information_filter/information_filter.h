#ifndef _INFORMATION_FILTER_SOLVER_H_
#define _INFORMATION_FILTER_SOLVER_H_

#include "basic_type.h"
#include "inverse_filter.h"

namespace slam_solver {

/* Class Basic Information Filter Declaration. */
template <typename Scalar>
class InformationFilterDynamic : public InverseFilter<Scalar, InformationFilterDynamic<Scalar>> {

public:
    InformationFilterDynamic()
        : InverseFilter<Scalar, InformationFilterDynamic<Scalar>>() {}
    virtual ~InformationFilterDynamic() = default;

    bool PropagateInformationImpl();
    bool UpdateStateAndInformationImpl(const TMat<Scalar> &observation = TMat<Scalar>::Zero(1, 1));

    // Reference for member variables.
    TVec<Scalar> &x() { return x_; }
    TMat<Scalar> &I() { return I_; }
    TVec<Scalar> &predict_x() { return predict_x_; }
    TMat<Scalar> &predict_I() { return predict_I_; }
    TMat<Scalar> &F() { return F_; }
    TMat<Scalar> &H() { return H_; }
    TMat<Scalar> &inverse_Q() { return inverse_Q_; }
    TMat<Scalar> &inverse_R() { return inverse_R_; }

    // Const reference for member variables.
    const TVec<Scalar> &x() const { return x_; }
    const TMat<Scalar> &I() const { return I_; }
    const TVec<Scalar> &predict_x() const { return predict_x_; }
    const TMat<Scalar> &predict_I() const { return predict_I_; }
    const TMat<Scalar> &F() const { return F_; }
    const TMat<Scalar> &H() const { return H_; }
    const TMat<Scalar> &inverse_Q() const { return inverse_Q_; }
    const TMat<Scalar> &inverse_R() const { return inverse_R_; }

private:
    TVec<Scalar> x_ = TVec<Scalar>::Zero(1, 1);
    TMat<Scalar> I_ = TMat<Scalar>::Zero(1, 1);

    TVec<Scalar> predict_x_ = TMat<Scalar>::Zero(1, 1);
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
class InformationFilterStatic : public InverseFilter<Scalar, InformationFilterStatic<Scalar, StateSize, ObserveSize>> {

    static_assert(StateSize > 0 && ObserveSize > 0, "Size of state and observe must be larger than 0.");

public:
    InformationFilterStatic()
        : InverseFilter<Scalar, InformationFilterStatic<Scalar, StateSize, ObserveSize>>() {}
    virtual ~InformationFilterStatic() = default;

    bool PropagateInformationImpl();
    bool UpdateStateAndInformationImpl(const TMat<Scalar> &observation = TMat<Scalar>::Zero(1, 1));

    // Reference for member variables.
    TVec<Scalar, StateSize> &x() { return x_; }
    TMat<Scalar, StateSize, StateSize> &I() { return I_; }
    TVec<Scalar, StateSize> &predict_x() { return predict_x_; }
    TMat<Scalar, StateSize, StateSize> &predict_I() { return predict_I_; }
    TMat<Scalar, StateSize, StateSize> &F() { return F_; }
    TMat<Scalar, ObserveSize, StateSize> &H() { return H_; }
    TMat<Scalar, StateSize, StateSize> &inverse_Q() { return inverse_Q_; }
    TMat<Scalar, ObserveSize, ObserveSize> &inverse_R() { return inverse_R_; }

    // Const reference for member variables.
    const TVec<Scalar, StateSize> &x() const { return x_; }
    const TMat<Scalar, StateSize, StateSize> &I() const { return I_; }
    const TVec<Scalar, StateSize> &predict_x() const { return predict_x_; }
    const TMat<Scalar, StateSize, StateSize> &predict_I() const { return predict_I_; }
    const TMat<Scalar, StateSize, StateSize> &F() const { return F_; }
    const TMat<Scalar, ObserveSize, StateSize> &H() const { return H_; }
    const TMat<Scalar, StateSize, StateSize> &inverse_Q() const { return inverse_Q_; }
    const TMat<Scalar, ObserveSize, ObserveSize> &inverse_R() const { return inverse_R_; }

private:
    TVec<Scalar, StateSize> x_ = TMat<Scalar>::Zero(1, 1);
    TMat<Scalar, StateSize, StateSize> I_ = TMat<Scalar>::Zero(1, 1);

    TVec<Scalar, StateSize> predict_x_ = TMat<Scalar>::Zero(1, 1);
    TMat<Scalar, StateSize, StateSize> predict_I_ = TMat<Scalar>::Zero(1, 1);

    // Process function F and measurement function H.
    TMat<Scalar, StateSize, StateSize> F_ = TMat<Scalar>::Identity(1, 1);
    TMat<Scalar, ObserveSize, StateSize> H_ = TMat<Scalar>::Identity(1, 1);

    // Process noise Q and measurement noise R.
    TMat<Scalar, StateSize, StateSize> inverse_Q_ = TMat<Scalar>::Zero(1, 1);
    TMat<Scalar, ObserveSize, ObserveSize> inverse_R_ = TMat<Scalar>::Zero(1, 1);
};

/* Class Basic Information Filter Definition. */
template <typename Scalar, int32_t StateSize, int32_t ObserveSize>
bool InformationFilterStatic<Scalar, StateSize, ObserveSize>::PropagateInformationImpl() {
    const TMat<Scalar, StateSize, StateSize> F_t = F_.transpose();
    predict_I_ = inverse_Q_ - inverse_Q_ * F_ * (I_ + F_t * inverse_Q_ * F_).inverse() * F_t * inverse_Q_;
    return true;
}

template <typename Scalar, int32_t StateSize, int32_t ObserveSize>
bool InformationFilterStatic<Scalar, StateSize, ObserveSize>::UpdateStateAndInformationImpl(const TMat<Scalar> &observation) {
    const TMat<Scalar, StateSize, ObserveSize> H_t = H_.transpose();

    // Update information.
    I_ = predict_I_ + H_t * inverse_R_ * H_;

    // Compute kalman gain.
    const TMat<Scalar, StateSize, ObserveSize> K_ = I_.inverse() * H_t * inverse_R_;

    // Update new state.
    const TVec<Scalar, ObserveSize> v_ = observation - H_ * predict_x_;
    x_ = predict_x_ + K_ * v_;
    return true;
}

}  // namespace slam_solver

#endif  // end of _INFORMATION_FILTER_SOLVER_H_
