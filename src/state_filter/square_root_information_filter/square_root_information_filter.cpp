#include "square_root_information_filter.h"

namespace SLAM_SOLVER {

/* Specialized Template Class Declaration. */
template class SquareRootInformationFilterDynamic<float>;
template class SquareRootInformationFilterDynamic<double>;

/* Class Square Root Error State Informaion Filter Definition. */
template <typename Scalar>
bool SquareRootInformationFilterDynamic<Scalar>::PropagateNominalStateImpl(const TVec<Scalar> &parameters) {
    // For error state filter, nominal state propagation should not only use F_.
    return true;
}

template <typename Scalar>
bool SquareRootInformationFilterDynamic<Scalar>::PropagateInformationImpl() {
    // TODO: something wrong here.
    const int32_t state_size = kesi_t_.rows();
    const int32_t double_state_size = state_size << 1;
    if (extend_rho_t_.rows() != double_state_size) {
        extend_rho_t_.setZero(double_state_size, state_size);
    }

    const TMat<Scalar> F_inv = F_.inverse();
    const TMat<Scalar> F_t_inv = F_inv.transpose();

    /* extend_rho_t_ = [ kesi_t * F.inv ]
                       [   Q.inv.t/2    ] */
    extend_rho_t_.template block(0, 0, state_size, state_size) = kesi_t_ * F_inv;
    extend_rho_t_.template block(state_size, 0, state_size, state_size) = inv_sqrt_Q_t_;

    // After QR decomposing of extend_rho_t_, the top matrix of the upper triangular matrix becomes rho.
    Eigen::HouseholderQR<TMat<Scalar>> qr_solver(extend_rho_t_);
    extend_rho_t_ = qr_solver.matrixQR().template triangularView<Eigen::Upper>();
    const TMat<Scalar> rho = extend_rho_t_.template block(0, 0, state_size, state_size);

    // Compute predict kesi_t.
    predict_kesi_t_ = F_t_inv * kesi_t_.transpose() - F_t_inv * kesi_t_.transpose() * kesi_t_ * F_inv * (
        rho * rho.transpose() + inv_sqrt_Q_t_ * rho.transpose()
    ).inverse() * F_t_inv * kesi_t_.transpose();
    return true;
}

template <typename Scalar>
bool SquareRootInformationFilterDynamic<Scalar>::UpdateStateAndInformationImpl(const TMat<Scalar> &residual) {
    // TODO: something wrong here.
    const int32_t state_size = kesi_t_.rows();
    const int32_t double_state_size = state_size << 1;
    if (extend_predict_kesi_t_.rows() != double_state_size) {
        extend_predict_kesi_t_.setZero(double_state_size, state_size);
    }
    const TMat<Scalar> kesi = kesi_t_.transpose();

    /* extend_predict_kesi_t_ = [ predict_kesi_t_ ]
                                [  R.inv.t/2 * H  ] */
    extend_predict_kesi_t_.template block(0, 0, state_size, state_size) = predict_kesi_t_;
    extend_predict_kesi_t_.template block(state_size, 0, state_size, state_size) = inv_sqrt_R_t_ * H_;

    // After QR decomposing of extend_predict_kesi_t_, the top matrix of the upper triangular matrix becomes kesi_t_.
    Eigen::HouseholderQR<TMat<Scalar>> qr_solver(extend_predict_kesi_t_);
    extend_predict_kesi_t_ = qr_solver.matrixQR().template triangularView<Eigen::Upper>();
    kesi_t_ = extend_predict_kesi_t_.template block(0, 0, state_size, state_size);

    // Compute kalman gain.
    const TMat<Scalar> K_ = (kesi * kesi_t_).inverse() * H_.transpose() * inv_sqrt_R_t_ * inv_sqrt_R_t_;

    // Update error state.
    dx_ = K_ * residual;
    return true;
}

}
