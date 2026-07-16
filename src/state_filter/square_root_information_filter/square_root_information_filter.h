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
 * Conventions:
 *   I = W^T * W  (information matrix from square root factor)
 *   b = W * dx   (information vector: transformed error state)
 *   dx = W^{-1} * b
 *   P = I^{-1} = (W^T * W)^{-1}
 *
 * Algorithm Flow (Dyer-McReynolds):
 * 1. Predict (Propagate):
 *    - dx = 0, b = 0 (reset error state info)
 *    - Matrix Augmentation: A = [    W_{k-1}    ,       0      | b_{k-1} ]
 *                                [ -(L_Q^{-1}) ,   (L_Q^{-1})  |   0     ]
 *      where L_Q is the lower Cholesky factor of Q.
 *      (inv_sqrt_Q_t_ = L_Q^{-T}, so the code uses inv_sqrt_Q_t_.transpose()
 *       to get L_Q^{-1} which satisfies (L_Q^{-1})^T * L_Q^{-1} = Q^{-1}.)
 *    - QR decomposition on A: Q * A = [ R11 , R12 | b* ; 0 , W_pre | b_pre ]
 *    - W_pre is the square root of the predicted information matrix:
 *      W_pre^T * W_pre = (F * (W_{k-1}^T*W_{k-1})^{-1} * F^T + Q)^{-1}
 * 2. Update:
 *    - Matrix Augmentation: B = [ W_pre , b_pre ; (L_R^{-1}) * H , (L_R^{-1}) * residual ]
 *      where L_R is the lower Cholesky factor of R.
 *      (inv_sqrt_R_t_ = L_R^{-T}, transposed to get L_R^{-1} satisfying
 *       (L_R^{-1})^T * L_R^{-1} = R^{-1}.)
 *    - QR decomposition on B: Q * B = [ W_new , b_new ; 0 , residual_new ]
 *    - dx = W_new^{-1} * b_new
 *
 * Variables:
 * - dx: Error state vector
 * - W: Upper-triangular square root of information matrix (I = W^T * W)
 * - b: Transformed information vector (b = W * dx)
 * - inv_sqrt_Q_t: Q^{-T/2}, satisfies inv_sqrt_Q_t * inv_sqrt_Q_t^T = Q^{-1}.
 *                 Internally the code uses inv_sqrt_Q_t^T (= L_Q^{-1}) in the
 *                 augmented matrix so that (L_Q^{-1})^T * L_Q^{-1} = Q^{-1}.
 * - inv_sqrt_R_t: R^{-T/2}, satisfies inv_sqrt_R_t * inv_sqrt_R_t^T = R^{-1}.
 *                 Similarly transposed internally so that (L_R^{-1})^T * L_R^{-1} = R^{-1}.
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
    TMat<Scalar> &null_space() { return null_space_; }

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
    const TMat<Scalar> &null_space() const { return null_space_; }

private:
    TVec<Scalar> dx_ = TVec<Scalar>::Zero(1, 1);
    // I = W^T * W.
    TMat<Scalar> W_ = TMat<Scalar>::Zero(1, 1);
    // b = W * dx.
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

    // Null space matrix for projecting the state update.
    // When set (cols > 0), dx is projected as
    // dx_proj = (I - N * (N^T*N)^{-1} * N^T) * dx,
    // so that states in the column space of N are unaffected by the observation.
    TMat<Scalar> null_space_ = TMat<Scalar>::Zero(0, 0);
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
    TMat<Scalar, StateSize, Eigen::Dynamic> &null_space() { return null_space_; }

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
    const TMat<Scalar, StateSize, Eigen::Dynamic> &null_space() const { return null_space_; }

private:
    TVec<Scalar, StateSize> dx_ = TVec<Scalar, StateSize>::Zero();
    // I = W^T * W.
    TMat<Scalar, StateSize, StateSize> W_ = TMat<Scalar, StateSize, StateSize>::Zero();
    // b = W * dx.
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

    // Null space matrix for projecting the state update.
    // When set (cols > 0), dx is projected as
    // dx_proj = (I - N * (N^T*N)^{-1} * N^T) * dx,
    // so that states in the column space of N are unaffected by the observation.
    TMat<Scalar, StateSize, Eigen::Dynamic> null_space_;
};

/* Class Square Root Error State Information Filter Definition. */
template <typename Scalar, int32_t StateSize, int32_t ObserveSize>
bool SquareRootInformationFilterStatic<Scalar, StateSize, ObserveSize>::PropagateInformationImpl() {
    dx_.setZero();
    b_.setZero();
    const int32_t state_size = W_.rows();

    /* A = [      W_             0      | b ]
           [ -inv_sqrt_Q_t^{T} * F  inv_sqrt_Q_t^{T} | 0 ]
       where (inv_sqrt_Q_t^{T})^T * inv_sqrt_Q_t^{T} = Q^{-1}.
       After QR:
       Q * A = [ R11   R12   | b* ]
               [  0     Wk   | bk ]
       Then: I_pred = Wk^T * Wk, and bk = Wk * dx_pred (= 0 for error state). */
    A_.setZero();
    A_.template block<StateSize, StateSize>(0, 0) = W_;
    A_.template block<StateSize, 1>(0, state_size << 1) = b_;
    A_.template block<StateSize, StateSize>(StateSize, 0) = -inv_sqrt_Q_t_.transpose() * F_;
    A_.template block<StateSize, StateSize>(StateSize, StateSize) = inv_sqrt_Q_t_.transpose();

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

    /* B = [ predict_W        | predict_b ]
           [ inv_sqrt_R_t^{T} * H | inv_sqrt_R_t^{T} * residual ]
       where (inv_sqrt_R_t^{T})^T * inv_sqrt_R_t^{T} = R^{-1}.
       Then QR on B:
       Q * B = [ W_new | b_new ]
               [   0   |   r   ]
       Then: I_new = W_new^T * W_new, b_new = W_new * dx_new, dx_new = W_new^{-1} * b_new. */
    B_.setZero();
    B_.template block(0, 0, state_size, state_size) = predict_W_;
    B_.template block(0, state_size, state_size, 1) = predict_b_;
    B_.template block(state_size, 0, measure_size, state_size) = inv_sqrt_R_t_.transpose() * H_;
    B_.template block(state_size, state_size, measure_size, 1) = inv_sqrt_R_t_.transpose() * residual;

    // After QR decomposing of B_, the top left block is new W_.
    Eigen::HouseholderQR<TMat<Scalar, StateSize + ObserveSize, StateSize + 1>> qr_solver(B_);
    TMat<Scalar, StateSize + ObserveSize, StateSize + 1> R_upper = qr_solver.matrixQR().template triangularView<Eigen::Upper>();

    W_ = R_upper.template block(0, 0, state_size, state_size);
    b_ = R_upper.template block(0, state_size, state_size, 1);

    // Update error state.
    dx_ = W_.template triangularView<Eigen::Upper>().solve(b_);

    // Project dx using null space (if set).
    // dx_proj = (I - N * (N^T*N)^{-1} * N^T) * dx
    // States in the column space of null_space_ will not be affected by the observation.
    if (null_space_.cols() > 0) {
        const TMat<Scalar> N = null_space_;
        const TMat<Scalar> NtN = N.transpose() * N;
        dx_ -= N * NtN.ldlt().solve(N.transpose() * dx_);

        // Recompute W and b to be consistent with the null space projection.
        // (1) W_inv  = predict_W^{-1}                          (back-substitution)
        // (2) P_pred = W_inv * W_inv^T
        // (3) K      = P_pred * H^T * (H*P_pred*H^T + R)^{-1}
        // (4) K_proj = P_N * K
        // (5) P_new  = (I - K_proj*H) * P_pred * (I - K_proj*H)^T + K_proj*R*K_proj^T
        // (6) I_new  = P_new^{-1}
        // (7) W_new  = LLT(I_new).matrixU()                    (upper Cholesky of I_new)
        // (8) b_new  = W_new * dx_proj
        //
        // This avoids forming predict_I = predict_W^T * predict_W (the quadratic product
        // on the information side).  Step (6)–(7) are unavoidable because no orthogonal
        // transformation can turn P_new^{-1} into an upper-triangular factor without
        // first forming I_new — the two Cholesky orientations (P_new = L*L^T vs
        // I_new = U^T*U) give structurally different triangular matrices.
        const TMat<Scalar, StateSize, StateSize> W_inv =
            predict_W_.template triangularView<Eigen::Upper>().solve(TMat<Scalar, StateSize, StateSize>::Identity());
        const TMat<Scalar, StateSize, StateSize> P_pred = W_inv * W_inv.transpose();
        const TMat<Scalar, StateSize, ObserveSize> H_t = H_.transpose();

        // R from inv_sqrt_R_t:  inv_sqrt_R_t * inv_sqrt_R_t^T = R^{-1}
        const TMat<Scalar, ObserveSize, ObserveSize> inverse_R = inv_sqrt_R_t_ * inv_sqrt_R_t_.transpose();
        const TMat<Scalar, ObserveSize, ObserveSize> R_mat = inverse_R.ldlt().solve(TMat<Scalar, ObserveSize, ObserveSize>::Identity());

        // K = P_pred * H^T * (H * P_pred * H^T + R)^{-1}
        const TMat<Scalar, ObserveSize, ObserveSize> S = H_ * P_pred * H_t + R_mat;
        const TMat<Scalar, StateSize, ObserveSize> K = P_pred * H_t * S.ldlt().solve(TMat<Scalar, ObserveSize, ObserveSize>::Identity());

        // K_proj = P_N * K
        const TMat<Scalar, StateSize, ObserveSize> K_proj = K - N * NtN.ldlt().solve(N.transpose() * K);

        // P_new via Joseph form
        const TMat<Scalar, StateSize, StateSize> I_mat = TMat<Scalar, StateSize, StateSize>::Identity();
        const TMat<Scalar, StateSize, StateSize> I_KH = I_mat - K_proj * H_;
        const TMat<Scalar, StateSize, StateSize> P_new = I_KH * P_pred * I_KH.transpose() + K_proj * R_mat * K_proj.transpose();

        // I_new = P_new^{-1}  →  W_new = upper Cholesky of I_new
        const TMat<Scalar, StateSize, StateSize> I_new = P_new.ldlt().solve(TMat<Scalar, StateSize, StateSize>::Identity());
        Eigen::LLT<TMat<Scalar, StateSize, StateSize>> llt_i(I_new);
        W_ = llt_i.matrixU();
        b_ = W_ * dx_;
    }

    return true;
}

}  // namespace slam_solver

#endif  // end of _SQUARE_ROOT_INFORMATION_FILTER_SOLVER_H_
