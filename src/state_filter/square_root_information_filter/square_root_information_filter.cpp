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
           [ -sqrt(Q).inv * F  sqrt(Q).inv | 0 ]
       After QR:
       T * A = [ R11   R12   | b* ]
               [  0     Wk   | bk ]
       Note: We must put x_{k-1} in the first n columns to eliminate it. */
    A_.setZero();
    A_.template block(0, 0, state_size, state_size) = W_;
    A_.template block(0, state_size << 1, state_size, 1) = b_;
    A_.template block(state_size, 0, state_size, state_size) = -inv_sqrt_Q_t_ * F_;
    A_.template block(state_size, state_size, state_size, state_size) = inv_sqrt_Q_t_;

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

    /* B = [ predict_W       | predict_b ]
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
    Eigen::HouseholderQR<TMat<Scalar>> qr_solver(B_);
    TMat<Scalar> R_upper = qr_solver.matrixQR().template triangularView<Eigen::Upper>();

    W_ = R_upper.template block(0, 0, state_size, state_size);
    b_ = R_upper.template block(0, state_size, state_size, 1);

    // Update error state.
    dx_ = W_.template triangularView<Eigen::Upper>().solve(b_);

    return true;
}

}  // namespace slam_solver
