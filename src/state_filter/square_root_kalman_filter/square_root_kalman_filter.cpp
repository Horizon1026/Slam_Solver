#include "square_root_kalman_filter.h"
#include "iostream"

namespace slam_solver {

/* Specialized Template Class Declaration. */
template class SquareRootKalmanFilterDynamic<float>;
template class SquareRootKalmanFilterDynamic<double>;

/* Class Square Root Error State Kalman Filter Definition. */
template <typename Scalar>
bool SquareRootKalmanFilterDynamic<Scalar>::PropagateCovarianceImpl() {
    dx_.setZero();
    const int32_t state_size = S_t_.cols();
    const int32_t extend_size = S_t_.rows() + state_size;
    if (extend_predict_S_t_.rows() != extend_size) {
        extend_predict_S_t_.setZero(extend_size, state_size);
    }

    /*  extend_predict_S_t_ = [ S.t * F.t ]
                              [   Q.t/2   ] */
    extend_predict_S_t_.template block(0, 0, S_t_.rows(), state_size) = S_t_ * F_.transpose();
    extend_predict_S_t_.template block(S_t_.rows(), 0, state_size, state_size) = sqrt_Q_t_;

    // After QR decomposing of extend_predict_S_t_, the top matrix of the upper triangular matrix becomes predict_S_t_.
    Eigen::HouseholderQR<TMat<Scalar>> qr_solver(extend_predict_S_t_);
    predict_S_t_ = qr_solver.matrixQR().template block(0, 0, state_size, state_size).template triangularView<Eigen::Upper>();
    return true;
}

template <typename Scalar>
bool SquareRootKalmanFilterDynamic<Scalar>::UpdateStateAndCovarianceImpl(const TMat<Scalar> &residual) {
    const int32_t state_size = predict_S_t_.cols();
    const int32_t obv_size = H_.rows();
    if (residual.rows() != obv_size || H_.cols() != state_size) {
        return false;
    }

    const int32_t full_size = state_size + obv_size;
    if (M_.rows() != full_size || M_.cols() != full_size) {
        M_.setZero(full_size, full_size);
    }

    // Construct matrix M, do QR decompose on it.
    /*  M = [ R.t/2             0    ] = T * [ (H * pre_P * H.t + R).t/2  hat_K.t ]
            [ pre_S.t * H.t  pre_S.t ]       [             0                S.t   ]
        T is a exist unit orthogonal matrix. */
    M_.template block(0, 0, obv_size, obv_size) = sqrt_R_t_;
    M_.template block(0, obv_size, obv_size, state_size).setZero();
    M_.template block(obv_size, 0, state_size, obv_size) = predict_S_t_ * H_.transpose();
    M_.template block(obv_size, obv_size, state_size, state_size) = predict_S_t_;
    Eigen::HouseholderQR<TMat<Scalar>> qr_solver(M_);
    TMat<Scalar> R_upper = qr_solver.matrixQR().template triangularView<Eigen::Upper>();

    // Compute Kalman gain.
    // hat_K = (H * pre_P * H.t + R).t/2 * K.
    const TMat<Scalar> sqrt_S_t = R_upper.template block(0, 0, obv_size, obv_size);
    const TMat<Scalar> hat_K_t = R_upper.template block(0, obv_size, obv_size, state_size);
    const TMat<Scalar> K_ = hat_K_t.transpose() * sqrt_S_t.template triangularView<Eigen::Upper>().solve(TMat<Scalar>::Identity(obv_size, obv_size));

    // Update error state.
    dx_ = K_ * residual;

    // Update covariance of new state.
    switch (options_.kMethod) {
        default:
        case StateCovUpdateMethod::kSimple:
        case StateCovUpdateMethod::kFull:
            S_t_ = R_upper.template block(obv_size, obv_size, state_size, state_size);
            break;
    }

    return true;
}

}  // namespace slam_solver
