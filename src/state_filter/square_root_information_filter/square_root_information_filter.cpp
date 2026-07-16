#include "square_root_information_filter.h"

namespace slam_solver {

/* Specialized Template Class Declaration. */
template class SquareRootInformationFilterDynamic<float>;
template class SquareRootInformationFilterDynamic<double>;

/* Class Square Root Error State Informaion Filter Definition. */
template <typename Scalar>
bool SquareRootInformationFilterDynamic<Scalar>::PropagateInformationImpl() {
    dx_.setZero();
    b_.setZero();
    const int32_t state_size = W_.rows();
    const int32_t double_state_size = state_size << 1;
    if (A_.rows() != double_state_size || A_.cols() != double_state_size + 1) {
        A_.setZero(double_state_size, double_state_size + 1);
    }
    if (predict_b_.rows() != state_size) {
        predict_b_.setZero(state_size, 1);
    }

    /* A = [      W_             0      | b ]
           [ -inv_sqrt_Q_t^{T} * F  inv_sqrt_Q_t^{T} | 0 ]
       where (inv_sqrt_Q_t^{T})^T * inv_sqrt_Q_t^{T} = Q^{-1}.
       After QR:
       Q * A = [ R11   R12   | b* ]
               [  0     Wk   | bk ]
       Then: I_pred = Wk^T * Wk, and bk = Wk * dx_pred (= 0 for error state). */
    A_.setZero();
    A_.template block(0, 0, state_size, state_size) = W_;
    A_.template block(0, state_size << 1, state_size, 1) = b_;
    A_.template block(state_size, 0, state_size, state_size) = -inv_sqrt_Q_t_.transpose() * F_;
    A_.template block(state_size, state_size, state_size, state_size) = inv_sqrt_Q_t_.transpose();

    // After QR decomposing of A_, the bottom right N x N block is predict_W_,
    // and the bottom right N x 1 block of the last column is predict_b_.
    Eigen::HouseholderQR<TMat<Scalar>> qr_solver(A_);
    TMat<Scalar> R_upper = qr_solver.matrixQR().template triangularView<Eigen::Upper>();

    predict_W_ = R_upper.template block(state_size, state_size, state_size, state_size);
    predict_b_ = R_upper.template block(state_size, state_size << 1, state_size, 1);

    return true;
}

template <typename Scalar>
bool SquareRootInformationFilterDynamic<Scalar>::UpdateStateAndInformationImpl(const TMat<Scalar> &residual) {
    const int32_t state_size = W_.rows();
    const int32_t measure_size = inv_sqrt_R_t_.rows();
    if (B_.rows() != state_size + measure_size || B_.cols() != state_size + 1) {
        B_.setZero(state_size + measure_size, state_size + 1);
    }

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
    Eigen::HouseholderQR<TMat<Scalar>> qr_solver(B_);
    TMat<Scalar> R_upper = qr_solver.matrixQR().template triangularView<Eigen::Upper>();

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
        const int32_t state_size = W_.rows();
        const int32_t meas_size = inv_sqrt_R_t_.rows();

        const TMat<Scalar> W_inv = predict_W_.template triangularView<Eigen::Upper>().solve(TMat<Scalar>::Identity(state_size, state_size));
        const TMat<Scalar> P_pred = W_inv * W_inv.transpose();
        const TMat<Scalar> H_t = H_.transpose();

        const TMat<Scalar> inverse_R = inv_sqrt_R_t_ * inv_sqrt_R_t_.transpose();
        const TMat<Scalar> R_mat = inverse_R.ldlt().solve(TMat<Scalar>::Identity(meas_size, meas_size));

        const TMat<Scalar> S = H_ * P_pred * H_t + R_mat;
        const TMat<Scalar> K = P_pred * H_t * S.ldlt().solve(TMat<Scalar>::Identity(meas_size, meas_size));

        const TMat<Scalar> K_proj = K - N * NtN.ldlt().solve(N.transpose() * K);

        const TMat<Scalar> I_mat = TMat<Scalar>::Identity(state_size, state_size);
        const TMat<Scalar> I_KH = I_mat - K_proj * H_;
        const TMat<Scalar> P_new = I_KH * P_pred * I_KH.transpose() + K_proj * R_mat * K_proj.transpose();

        // I_new = P_new^{-1}  →  W_new = upper Cholesky of I_new
        const TMat<Scalar> I_new = P_new.ldlt().solve(TMat<Scalar>::Identity(state_size, state_size));
        Eigen::LLT<TMat<Scalar>> llt_i(I_new);
        W_ = llt_i.matrixU();
        b_ = W_ * dx_;
    }

    return true;
}

}  // namespace slam_solver
