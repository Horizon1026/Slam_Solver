#include "square_root_information_filter.h"

namespace SLAM_SOLVER {

/* Specialized Template Class Declaration. */
template class SquareRootInformationFilterDynamic<float>;
template class SquareRootInformationFilterDynamic<double>;

/* Class Square Root Error State Informaion Filter Definition. */
template <typename Scalar>
bool SquareRootInformationFilterDynamic<Scalar>::PropagateInformationImpl() {
    const int32_t state_size = W_.rows();
    const int32_t double_state_size = state_size << 1;
    if (A_.rows() != double_state_size) {
        A_.setZero(double_state_size, state_size);
    }

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

template <typename Scalar>
bool SquareRootInformationFilterDynamic<Scalar>::UpdateStateAndInformationImpl(const TMat<Scalar> &residual) {
    const int32_t state_size = W_.rows();
    const int32_t measure_size = inv_sqrt_R_t_.rows();
    if (B_.rows() != state_size + measure_size) {
        B_.setZero(state_size + measure_size, state_size + 1);
    }

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

}
