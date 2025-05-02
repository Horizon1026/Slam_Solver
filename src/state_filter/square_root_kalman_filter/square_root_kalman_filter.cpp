#include "square_root_kalman_filter.h"
#include "iostream"

namespace SLAM_SOLVER {

/* Specialized Template Class Declaration. */
template class SquareRootKalmanFilterDynamic<float>;
template class SquareRootKalmanFilterDynamic<double>;

/* Class Square Root Error State Kalman Filter Definition. */
template <typename Scalar>
bool SquareRootKalmanFilterDynamic<Scalar>::PropagateNominalStateImpl(const TVec<Scalar> &parameters) {
    // For error state filter, nominal state propagation should not only use F_.
    return true;
}

template <typename Scalar>
bool SquareRootKalmanFilterDynamic<Scalar>::PropagateCovarianceImpl() {
    const int32_t state_size = S_t_.rows();
    const int32_t double_state_size = state_size << 1;
    if (extend_predict_S_t_.rows() != double_state_size) {
        extend_predict_S_t_.setZero(double_state_size, state_size);
    }

    /*  extend_predict_S_t_ = [ S.t * F.t ]
                              [   Q.t/2   ] */
    extend_predict_S_t_.template block(0, 0, state_size, state_size) = S_t_ * F_.transpose();
    extend_predict_S_t_.template block(state_size, 0, state_size, state_size) = sqrt_Q_t_;

    // After QR decomposing of extend_predict_S_t_, the top matrix of the upper triangular matrix becomes predict_S_t_.
    Eigen::HouseholderQR<TMat<Scalar>> qr_solver(extend_predict_S_t_);
    extend_predict_S_t_ = qr_solver.matrixQR().template triangularView<Eigen::Upper>();
    predict_S_t_ = extend_predict_S_t_.template block(0, 0, state_size, state_size);
    return true;
}

template <typename Scalar>
bool SquareRootKalmanFilterDynamic<Scalar>::UpdateStateAndCovarianceImpl(const TMat<Scalar> &residual) {
    const int32_t state_size = predict_S_t_.rows();
    const int32_t obv_size = H_.rows();
    if (residual.rows() != obv_size) {
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
    M_ = qr_solver.matrixQR().template triangularView<Eigen::Upper>();

    // Commpute Kalman gain.
    // hat_K = (H * pre_P * H.t + R).t/2 * K.
    const TMat<Scalar> K_
        = M_.template block(0, obv_size, obv_size, state_size).transpose()
        * M_.template block(0, 0, obv_size, obv_size).inverse();

    // Update error state.
    dx_ = K_ * residual;

    // Update covariance of new state.
    switch (options_.kMethod) {
        default:
        case StateCovUpdateMethod::kSimple:
        case StateCovUpdateMethod::kFull:
            S_t_ = M_.template block(obv_size, obv_size, state_size, state_size);
            break;
    }

    return true;
}

}
