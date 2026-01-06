#ifndef _SQUARE_ROOT_INFORMATION_FILTER_SOLVER_H_
#define _SQUARE_ROOT_INFORMATION_FILTER_SOLVER_H_

#include "basic_type.h"
#include "inverse_filter.h"

namespace slam_solver {

/**
 * @brief Square Root Information Filter (SRIF)
 *
 * References:
 * - "Optimal State Estimation: Kalman, H Infinity, and Nonlinear Approaches", Dan Simon.
 * - "Factorization Methods for Discrete Sequential Estimation", Gerald J. Bierman.
 *
 * Algorithm Flow (Dyer-McReynolds):
 * 1. Predict (Propagate):
 *    - dx = 0, b = 0 (reset error state info)
 *    - Matrix Augmentation: A = [ Q^{-T/2} , -Q^{-T/2} * F , 0 ; 0 , W_{k-1} , b_{k-1} ]
 *    - QR decomposition on A: T * A = [ R11 , R12 , b* ; 0 , W_pre , b_pre ]
 * 2. Update:
 *    - Matrix Augmentation: B = [ W_pre , b_pre ; R^{-T/2} * H , R^{-T/2} * residual ]
 *    - QR decomposition on B: T * B = [ W_new , b_new ; 0 , residual_new ]
 *    - dx = W_new^-1 * b_new
 *
 * Variables:
 * - dx: Error state vector
 * - W: Square root of information matrix (I = W * W^T)
 * - b: Information vector (I * dx = b)
 * - inv_sqrt_Q_t: Square root of process noise info matrix (Q^-1 = inv_sqrt_Q * inv_sqrt_Q^T)
 * - inv_sqrt_R_t: Square root of measurement noise info matrix (R^-1 = inv_sqrt_R * inv_sqrt_R^T)
 */
template <typename Scalar>
class SquareRootInformationFilterDynamic: public InverseFilter<Scalar, SquareRootInformationFilterDynamic<Scalar>> {

public:
    SquareRootInformationFilterDynamic(): InverseFilter<Scalar, SquareRootInformationFilterDynamic<Scalar>>() {}
    virtual ~SquareRootInformationFilterDynamic() = default;

    bool PropagateInformationImpl();
    bool UpdateStateAndInformationImpl(const TMat<Scalar> &observation = TMat<Scalar>::Zero(1, 1));

    // Reference for member variables.
    TVec<Scalar> &dx() { return dx_; }
    TMat<Scalar> &W() { return W_; }
    TVec<Scalar> &b() { return b_; }
    TMat<Scalar> &F() { return F_; }
    TMat<Scalar> &H() { return H_; }
    TMat<Scalar> &inv_sqrt_Q_t() { return inv_sqrt_Q_t_; }
    TMat<Scalar> &inv_sqrt_R_t() { return inv_sqrt_R_t_; }
    TMat<Scalar> &predict_W() { return predict_W_; }
    TVec<Scalar> &predict_b() { return predict_b_; }

    // Const reference for member variables.
    const TVec<Scalar> &dx() const { return dx_; }
    const TMat<Scalar> &W() const { return W_; }
    const TVec<Scalar> &b() const { return b_; }
    const TMat<Scalar> &F() const { return F_; }
    const TMat<Scalar> &H() const { return H_; }
    const TMat<Scalar> &inv_sqrt_Q_t() const { return inv_sqrt_Q_t_; }
    const TMat<Scalar> &inv_sqrt_R_t() const { return inv_sqrt_R_t_; }
    const TMat<Scalar> &predict_W() const { return predict_W_; }
    const TVec<Scalar> &predict_b() const { return predict_b_; }

private:
    TVec<Scalar> dx_ = TVec<Scalar>::Zero(1, 1);
    // I is represent as W * W.t.
    TMat<Scalar> W_ = TMat<Scalar>::Zero(1, 1);
    TVec<Scalar> b_ = TVec<Scalar>::Zero(1, 1);

    // Process function F and measurement function H.
    TMat<Scalar> F_ = TMat<Scalar>::Identity(1, 1);
    TMat<Scalar> H_ = TMat<Scalar>::Zero(1, 1);

    // Process noise Q and measurement noise R.
    // Define Q^(-T/2) and R^(-T/2) here.
    TMat<Scalar> inv_sqrt_Q_t_ = TMat<Scalar>::Zero(1, 1);
    TMat<Scalar> inv_sqrt_R_t_ = TMat<Scalar>::Zero(1, 1);

    TMat<Scalar> A_ = TMat<Scalar>::Zero(1, 1);
    TMat<Scalar> B_ = TMat<Scalar>::Zero(1, 1);
    TMat<Scalar> predict_W_ = TMat<Scalar>::Zero(1, 1);
    TVec<Scalar> predict_b_ = TVec<Scalar>::Zero(1, 1);
};

/**
 * @brief Static Dimensional Square Root Information Filter (SRIF)
 * @tparam StateSize Dimension of the error state vector
 * @tparam ObserveSize Dimension of the measurement vector
 *
 * Algorithm and variables same as SquareRootInformationFilterDynamic.
 */
template <typename Scalar, int32_t StateSize, int32_t ObserveSize>
class SquareRootInformationFilterStatic: public InverseFilter<Scalar, SquareRootInformationFilterStatic<Scalar, StateSize, ObserveSize>> {

    static_assert(StateSize > 0 && ObserveSize > 0, "Size of state and observe must be larger than 0.");

public:
    SquareRootInformationFilterStatic(): InverseFilter<Scalar, SquareRootInformationFilterStatic<Scalar, StateSize, ObserveSize>>() {}
    virtual ~SquareRootInformationFilterStatic() = default;

    bool PropagateInformationImpl();
    bool UpdateStateAndInformationImpl(const TMat<Scalar> &observation = TMat<Scalar>::Zero(ObserveSize, 1));

    // Reference for member variables.
    TVec<Scalar, StateSize> &dx() { return dx_; }
    TMat<Scalar, StateSize, StateSize> &W() { return W_; }
    TVec<Scalar, StateSize> &b() { return b_; }
    TMat<Scalar, StateSize, StateSize> &F() { return F_; }
    TMat<Scalar, ObserveSize, StateSize> &H() { return H_; }
    TMat<Scalar, StateSize, StateSize> &inv_sqrt_Q_t() { return inv_sqrt_Q_t_; }
    TMat<Scalar, ObserveSize, ObserveSize> &inv_sqrt_R_t() { return inv_sqrt_R_t_; }
    TMat<Scalar, StateSize, StateSize> &predict_W() { return predict_W_; }
    TVec<Scalar, StateSize> &predict_b() { return predict_b_; }

    // Const reference for member variables.
    const TVec<Scalar, StateSize> &dx() const { return dx_; }
    const TMat<Scalar, StateSize, StateSize> &W() const { return W_; }
    const TVec<Scalar, StateSize> &b() const { return b_; }
    const TMat<Scalar, StateSize, StateSize> &F() const { return F_; }
    const TMat<Scalar, ObserveSize, StateSize> &H() const { return H_; }
    const TMat<Scalar, StateSize, StateSize> &inv_sqrt_Q_t() const { return inv_sqrt_Q_t_; }
    const TMat<Scalar, ObserveSize, ObserveSize> &inv_sqrt_R_t() const { return inv_sqrt_R_t_; }
    const TMat<Scalar, StateSize, StateSize> &predict_W() const { return predict_W_; }
    const TVec<Scalar, StateSize> &predict_b() const { return predict_b_; }

private:
    TVec<Scalar, StateSize> dx_ = TVec<Scalar, StateSize>::Zero();
    // I is represent as W * W.t.
    TMat<Scalar, StateSize, StateSize> W_ = TMat<Scalar, StateSize, StateSize>::Zero();
    TVec<Scalar, StateSize> b_ = TVec<Scalar, StateSize>::Zero();

    // Process function F and measurement function H.
    TMat<Scalar, StateSize, StateSize> F_ = TMat<Scalar, StateSize, StateSize>::Identity();
    TMat<Scalar, ObserveSize, StateSize> H_ = TMat<Scalar, ObserveSize, StateSize>::Zero();

    // Process noise Q and measurement noise R.
    // Define Q^(-T/2) and R^(-T/2) here.
    TMat<Scalar, StateSize, StateSize> inv_sqrt_Q_t_ = TMat<Scalar, StateSize, StateSize>::Zero();
    TMat<Scalar, ObserveSize, ObserveSize> inv_sqrt_R_t_ = TMat<Scalar, ObserveSize, ObserveSize>::Zero();

    TMat<Scalar, StateSize + StateSize, StateSize + StateSize + 1> A_ = TMat<Scalar, StateSize + StateSize, StateSize + StateSize + 1>::Zero();
    TMat<Scalar, StateSize + ObserveSize, StateSize + 1> B_ = TMat<Scalar, StateSize + ObserveSize, StateSize + 1>::Zero();
    TMat<Scalar, StateSize, StateSize> predict_W_ = TMat<Scalar, StateSize, StateSize>::Zero();
    TVec<Scalar, StateSize> predict_b_ = TVec<Scalar, StateSize>::Zero();
};

/* Class Square Root Error State Information Filter Definition. */
template <typename Scalar, int32_t StateSize, int32_t ObserveSize>
bool SquareRootInformationFilterStatic<Scalar, StateSize, ObserveSize>::PropagateInformationImpl() {
    dx_.setZero();
    b_.setZero();
    const int32_t state_size = W_.rows();

    /* A = [      W_             0      | b ]
           [ -sqrt(Q).inv * F  sqrt(Q).inv | 0 ]
       After QR:
       T * A = [ R11   R12   | b* ]
               [  0     Wk   | bk ]
       Note: We must put x_{k-1} in the first n columns to eliminate it. */
    A_.setZero();
    A_.template block<StateSize, StateSize>(0, 0) = W_;
    A_.template block<StateSize, 1>(0, state_size << 1) = b_;
    A_.template block<StateSize, StateSize>(StateSize, 0) = -inv_sqrt_Q_t_ * F_;
    A_.template block<StateSize, StateSize>(StateSize, StateSize) = inv_sqrt_Q_t_;

    // After QR decomposing of A_, the bottom right N x N block is predict_W_,
    // and the bottom right N x 1 block of the last column is predict_b_.
    Eigen::HouseholderQR<TMat<Scalar, StateSize + StateSize, StateSize + StateSize + 1>> qr_solver(A_);
    TMat<Scalar, StateSize + StateSize, StateSize + StateSize + 1> R_upper = qr_solver.matrixQR().template triangularView<Eigen::Upper>();

    predict_W_ = R_upper.template block<StateSize, StateSize>(StateSize, StateSize);
    predict_b_ = R_upper.template block<StateSize, 1>(StateSize, state_size << 1);

    return true;
}

template <typename Scalar, int32_t StateSize, int32_t ObserveSize>
bool SquareRootInformationFilterStatic<Scalar, StateSize, ObserveSize>::UpdateStateAndInformationImpl(const TMat<Scalar> &residual) {
    const int32_t state_size = W_.rows();
    const int32_t measure_size = inv_sqrt_R_t_.rows();

    /* B = [     predict_W           0        |  predict_b ]
           [ sqrt(R).inv * H  sqrt(R).inv * r |      0     ] -> Wait, actually:
       B = [ predict_W       | predict_b ]
           [ sqrt(R).inv * H | sqrt(R).inv * residual ]
       Then QR on B:
       T * B = [ W_new | b_new ]
               [   0   |   r   ] */
    B_.setZero();
    B_.template block(0, 0, state_size, state_size) = predict_W_;
    B_.template block(0, state_size, state_size, 1) = predict_b_;
    B_.template block(state_size, 0, measure_size, state_size) = inv_sqrt_R_t_ * H_;
    B_.template block(state_size, state_size, measure_size, 1) = inv_sqrt_R_t_ * residual;

    // After QR decomposing of B_, the top left block is new W_.
    Eigen::HouseholderQR<TMat<Scalar, StateSize + ObserveSize, StateSize + 1>> qr_solver(B_);
    TMat<Scalar, StateSize + ObserveSize, StateSize + 1> R_upper = qr_solver.matrixQR().template triangularView<Eigen::Upper>();

    W_ = R_upper.template block(0, 0, state_size, state_size);
    b_ = R_upper.template block(0, state_size, state_size, 1);

    // Update error state.
    dx_ = W_.template triangularView<Eigen::Upper>().solve(b_);

    return true;
}

}  // namespace slam_solver

#endif  // end of _SQUARE_ROOT_INFORMATION_FILTER_SOLVER_H_
