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
    M_.template block(obv_size, 0, state_size, obv_size).noalias() = predict_S_t_ * H_.transpose();
    M_.template block(obv_size, obv_size, state_size, state_size) = predict_S_t_;

    Eigen::HouseholderQR<Eigen::Ref<TMat<Scalar>>> qr_solver(M_);
    TMat<Scalar> R_upper = qr_solver.matrixQR().template triangularView<Eigen::Upper>();

    // Y = H * pre_P * H.t + R
    // so, sqrt_Y_t = (H * pre_P * H.t + R).t/2
    const TMat<Scalar> sqrt_Y = R_upper.template block(0, 0, obv_size, obv_size).transpose();
    TMat<Scalar> hat_K = R_upper.template block(0, obv_size, obv_size, state_size).transpose();
    // For kalman gain K, we have hat_K = K * sqrt_Y (Correct here. Reference book - 6.87 - is wrong.)
    // so, K = hat_K * sqrt_Y^{-1}

    // Null-space projection: K_proj = (I - N*(N^T*N)^{-1}*N^T) * K = K - N*(N^T*N)^{-1}*N^T * K
    // so project as hat_K -= N*(N^T*N)^{-1}*N^T * hat_K
    if (null_space_.cols() > 0) {
        const TMat<Scalar> NtN = null_space_.transpose() * null_space_;
        const TMat<Scalar> NtN_inv_Nt = NtN.ldlt().solve(null_space_.transpose());
        hat_K -= null_space_ * NtN_inv_Nt * hat_K;
    }

    // Compute projected Kalman gain K_t = (hat_K * sqrt_Y^-1) ^ {T}
    // dx = K * residual = hat_K * sqrt_Y^-1 * residual
    const TMat<Scalar> K_t = sqrt_Y.transpose().template triangularView<Eigen::Upper>().solve(hat_K.transpose());
    dx_.noalias() = K_t.transpose() * residual;

    // Update covariance of new state.
    if (null_space_.cols() > 0) {
        // B_t = [ S_up * (I - H_t*K_t) ]
        //       [    sqrt_R * K_t      ]  ((state+obsv) × state)
        // where S_up = predict_S_t_ is the stored S_t (upper tri).
        TMat<Scalar> B_t(full_size, state_size);
        const TMat<Scalar> I_HtKt = TMat<Scalar>::Identity(state_size, state_size) - H_.transpose() * K_t;
        B_t.topRows(state_size) = predict_S_t_ * I_HtKt;
        B_t.bottomRows(obv_size) = sqrt_R_t_ * K_t;
        Eigen::HouseholderQR<Eigen::Ref<TMat<Scalar>>> qr(B_t);
        S_t_ = qr.matrixQR().topRows(state_size).template triangularView<Eigen::Upper>();
    } else {
        S_t_ = R_upper.template block(obv_size, obv_size, state_size, state_size);
    }

    return true;
}

}  // namespace slam_solver
