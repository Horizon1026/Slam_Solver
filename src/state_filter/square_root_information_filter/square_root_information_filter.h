#ifndef _SQUARE_ROOT_INFORMATION_FILTER_SOLVER_H_
#define _SQUARE_ROOT_INFORMATION_FILTER_SOLVER_H_

#include "basic_type.h"
#include "inverse_filter.h"

namespace SLAM_SOLVER {

/* Class Square Root Error State Information Filter Declaration. */
template <typename Scalar>
class SquareRootInformationFilterDynamic : public InverseFilter<Scalar, SquareRootInformationFilterDynamic<Scalar>> {

public:
    SquareRootInformationFilterDynamic()
        : InverseFilter<Scalar, SquareRootInformationFilterDynamic<Scalar>>() {}
    virtual ~SquareRootInformationFilterDynamic() = default;

    bool PropagateInformationImpl();
    bool UpdateStateAndInformationImpl(const TMat<Scalar> &observation = TMat<Scalar>::Zero(1, 1));

    // Reference for member variables.
    TVec<Scalar> &dx() { return dx_; }
    TMat<Scalar> &W() { return W_; }
    TMat<Scalar> &F() { return F_; }
    TMat<Scalar> &H() { return H_; }
    TMat<Scalar> &inv_sqrt_Q_t() { return inv_sqrt_Q_t_; }
    TMat<Scalar> &inv_sqrt_R_t() { return inv_sqrt_R_t_; }
    TMat<Scalar> &predict_W() { return predict_W_; }

    // Const reference for member variables.
    const TVec<Scalar> &dx() const { return dx_; }
    const TMat<Scalar> &W() const { return W_; }
    const TMat<Scalar> &F() const { return F_; }
    const TMat<Scalar> &H() const { return H_; }
    const TMat<Scalar> &inv_sqrt_Q_t() const { return inv_sqrt_Q_t_; }
    const TMat<Scalar> &inv_sqrt_R_t() const { return inv_sqrt_R_t_; }
    const TMat<Scalar> &predict_W() const { return predict_W_; }

private:
    TVec<Scalar> dx_ = TVec<Scalar>::Zero(1, 1);
    // I is represent as W * W.t.
    TMat<Scalar> W_ = TMat<Scalar>::Zero(1, 1);

    // Process function F and measurement function H.
    TMat<Scalar> F_ = TMat<Scalar>::Identity(1, 1);
    TMat<Scalar> H_ = TMat<Scalar>::Identity(1, 1);

    // Process noise Q and measurement noise R.
    // Define Q^(-T/2) and R^(-T/2) here.
    TMat<Scalar> inv_sqrt_Q_t_ = TMat<Scalar>::Zero(1, 1);
    TMat<Scalar> inv_sqrt_R_t_ = TMat<Scalar>::Zero(1, 1);

    TMat<Scalar> A_ = TMat<Scalar>::Zero(1, 1);
    TMat<Scalar> B_ = TMat<Scalar>::Zero(1, 1);
    TMat<Scalar> predict_W_ = TMat<Scalar>::Zero(1, 1);
};

/* Class Square Root Error State Information Filter Declaration. */
template <typename Scalar, int32_t StateSize, int32_t ObserveSize>
class SquareRootInformationFilterStatic : public InverseFilter<Scalar, SquareRootInformationFilterStatic<Scalar, StateSize, ObserveSize>> {

    static_assert(StateSize > 0 && ObserveSize > 0, "Size of state and observe must be larger than 0.");

public:
    SquareRootInformationFilterStatic()
        : InverseFilter<Scalar, SquareRootInformationFilterStatic<Scalar, StateSize, ObserveSize>>() {}
    virtual ~SquareRootInformationFilterStatic() = default;

    bool PropagateInformationImpl();
    bool UpdateStateAndInformationImpl(const TMat<Scalar> &observation = TMat<Scalar>::Zero(ObserveSize, 1));

    // Reference for member variables.
    TVec<Scalar, StateSize> &dx() { return dx_; }
    TMat<Scalar, StateSize, StateSize> &W() { return W_; }
    TMat<Scalar, StateSize, StateSize> &F() { return F_; }
    TMat<Scalar, ObserveSize, StateSize> &H() { return H_; }
    TMat<Scalar, StateSize, StateSize> &inv_sqrt_Q_t() { return inv_sqrt_Q_t_; }
    TMat<Scalar, ObserveSize, ObserveSize> &inv_sqrt_R_t() { return inv_sqrt_R_t_; }
    TMat<Scalar, StateSize, StateSize> &predict_W() { return predict_W_; }

    // Const reference for member variables.
    const TVec<Scalar, StateSize> &dx() const { return dx_; }
    const TMat<Scalar, StateSize, StateSize> &W() const { return W_; }
    const TMat<Scalar, StateSize, StateSize> &F() const { return F_; }
    const TMat<Scalar, ObserveSize, StateSize> &H() const { return H_; }
    const TMat<Scalar, StateSize, StateSize> &inv_sqrt_Q_t() const { return inv_sqrt_Q_t_; }
    const TMat<Scalar, ObserveSize, ObserveSize> &inv_sqrt_R_t() const { return inv_sqrt_R_t_; }
    const TMat<Scalar, StateSize, StateSize> &predict_W() const { return predict_W_; }

private:
    TVec<Scalar, StateSize> dx_ = TVec<Scalar, StateSize>::Zero();
    // I is represent as W * W.t.
    TMat<Scalar, StateSize, StateSize> W_ = TMat<Scalar, StateSize, StateSize>::Zero();

    // Process function F and measurement function H.
    TMat<Scalar, StateSize, StateSize> F_ = TMat<Scalar, StateSize, StateSize>::Identity();
    TMat<Scalar, ObserveSize, StateSize> H_ = TMat<Scalar, ObserveSize, StateSize>::Identity();

    // Process noise Q and measurement noise R.
    // Define Q^(-T/2) and R^(-T/2) here.
    TMat<Scalar, StateSize, StateSize> inv_sqrt_Q_t_ = TMat<Scalar, StateSize, StateSize>::Zero();
    TMat<Scalar, ObserveSize, ObserveSize> inv_sqrt_R_t_ = TMat<Scalar, ObserveSize, ObserveSize>::Zero();

    TMat<Scalar, StateSize + StateSize, StateSize> A_ = TMat<Scalar, StateSize + StateSize, StateSize>::Zero();
    TMat<Scalar, StateSize + ObserveSize, StateSize + 1> B_ = TMat<Scalar, StateSize + ObserveSize, StateSize + 1>::Zero();
    TMat<Scalar, StateSize, StateSize> predict_W_ = TMat<Scalar, StateSize, StateSize>::Zero();
};

/* Class Square Root Error State Information Filter Definition. */
template <typename Scalar, int32_t StateSize, int32_t ObserveSize>
bool SquareRootInformationFilterStatic<Scalar, StateSize, ObserveSize>::PropagateInformationImpl() {
    const int32_t state_size = W_.rows();

    /* A = [        W_       ], T * A = [ predict_W_ ]
           [ sqrt(Q).inv * F ]          [     0      ]*/
    A_.template block(0, 0, state_size, state_size) = W_;
    A_.template block(state_size, 0, state_size, state_size) = inv_sqrt_Q_t_ * F_;

    // After QR decomposing of A_, the top left block is predict_W_.
    Eigen::HouseholderQR<TMat<Scalar>> qr_solver(A_);
    A_ = qr_solver.matrixQR().template triangularView<Eigen::Upper>();
    predict_W_ = A_.template block(0, 0, state_size, state_size);

    return true;
}

template <typename Scalar, int32_t StateSize, int32_t ObserveSize>
bool SquareRootInformationFilterStatic<Scalar, StateSize, ObserveSize>::UpdateStateAndInformationImpl(const TMat<Scalar> &residual) {
    const int32_t state_size = W_.rows();
    const int32_t measure_size = inv_sqrt_R_t_.rows();

    /* B = [     predict_W           0        ]
           [ sqrt(R).inv * H  sqrt(R).inv * r ] */
    B_.template block(0, 0, state_size, state_size) = predict_W_;
    B_.template block(state_size, 0, measure_size, state_size) = inv_sqrt_R_t_ * H_;
    B_.template block(state_size, state_size, measure_size, 1) = inv_sqrt_R_t_ * residual;

    // After QR decomposing of B_, the top left block is new W_.
    Eigen::HouseholderQR<TMat<Scalar>> qr_solver(B_);
    B_ = qr_solver.matrixQR().template triangularView<Eigen::Upper>();
    W_ = B_.template block(0, 0, state_size, state_size);

    // Update error state.
    const TVec<Scalar> error_b = B_.template block(0, state_size, state_size, 1);
    dx_ = W_.inverse() * error_b;

    return true;
}

}  // namespace SLAM_SOLVER

#endif  // end of _SQUARE_ROOT_INFORMATION_FILTER_SOLVER_H_
