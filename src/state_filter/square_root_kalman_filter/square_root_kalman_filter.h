#ifndef _SQUARE_ROOT_KALMAN_FILTER_SOLVER_H_
#define _SQUARE_ROOT_KALMAN_FILTER_SOLVER_H_

#include "basic_type.h"
#include "filter.h"

namespace slam_solver {

struct SquareRootKalmanFilterOptions {
    StateCovUpdateMethod kMethod = StateCovUpdateMethod::kSimple;
};

/**
 * @brief Square Root Kalman Filter (SRKF)
 *
 * References:
 * - "Optimal State Estimation: Kalman, H Infinity, and Nonlinear Approaches", Dan Simon.
 * - "Bierman, G. J. (1977). Factorization Methods for Discrete Sequential Estimation."
 *
 * Algorithm Flow:
 * 1. Predict:
 *    - dx = 0 (reset error state)
 *    - Matrix Augmentation: A = [ S_{k-1}^T * F^T ; Q^{T/2} ]
 *    - QR decomposition on A: T * A = [ S_pre^T ; 0 ]
 * 2. Update:
 *    - Matrix Augmentation: M = [ R^{T/2} , 0 ; S_pre^T * H^T , S_pre^T ]
 *    - QR decomposition on M: T * M = [ S_z^T , K_hat^T ; 0 , S_k^T ]
 *    - K = (K_hat * S_z^-1)^T
 *    - dx = K * residual
 *    - S_k^T is the new square root covariance.
 *
 * Variables:
 * - dx: Error state vector
 * - S_t: Cholesky factor of covariance (P = S * S^T, where S_t is S^T)
 * - F: State transition matrix
 * - H: Measurement Jacobian
 * - sqrt_Q_t: Cholesky factor of process noise (Q = sqrt_Q * sqrt_Q^T)
 * - sqrt_R_t: Cholesky factor of measurement noise (R = sqrt_R * sqrt_R^T)
 */
template <typename Scalar>
class SquareRootKalmanFilterDynamic: public Filter<Scalar, SquareRootKalmanFilterDynamic<Scalar>> {

public:
    SquareRootKalmanFilterDynamic(): Filter<Scalar, SquareRootKalmanFilterDynamic<Scalar>>() {}
    virtual ~SquareRootKalmanFilterDynamic() = default;

    bool PropagateCovarianceImpl();
    bool UpdateStateAndCovarianceImpl(const TMat<Scalar> &observation = TMat<Scalar>::Zero(1, 1));

    // Reference for member variables.
    SquareRootKalmanFilterOptions &options() { return options_; }
    TVec<Scalar> &dx() { return dx_; }
    TMat<Scalar> &S_t() { return S_t_; }
    TMat<Scalar> &predict_S_t() { return predict_S_t_; }
    TMat<Scalar> &F() { return F_; }
    TMat<Scalar> &H() { return H_; }
    TMat<Scalar> &sqrt_Q_t() { return sqrt_Q_t_; }
    TMat<Scalar> &sqrt_R_t() { return sqrt_R_t_; }

    // Const reference for member variables.
    const SquareRootKalmanFilterOptions &options() const { return options_; }
    const TVec<Scalar> &dx() const { return dx_; }
    const TMat<Scalar> &S_t() const { return S_t_; }
    const TMat<Scalar> &predict_S_t() const { return predict_S_t_; }
    const TMat<Scalar> &F() const { return F_; }
    const TMat<Scalar> &H() const { return H_; }
    const TMat<Scalar> &sqrt_Q_t() const { return sqrt_Q_t_; }
    const TMat<Scalar> &sqrt_R_t() const { return sqrt_R_t_; }

private:
    SquareRootKalmanFilterOptions options_;

    TVec<Scalar> dx_ = TVec<Scalar>::Zero(1, 1);
    // P is represent as P = S * S.t.
    TMat<Scalar> S_t_ = TMat<Scalar>::Zero(1, 1);

    // Process function F and measurement function H.
    TMat<Scalar> F_ = TMat<Scalar>::Identity(1, 1);
    TMat<Scalar> H_ = TMat<Scalar>::Identity(1, 1);

    // Process noise Q and measurement noise R.
    // Define Q^(T/2) and R^(T/2) here.
    TMat<Scalar> sqrt_Q_t_ = TMat<Scalar>::Zero(1, 1);
    TMat<Scalar> sqrt_R_t_ = TMat<Scalar>::Zero(1, 1);

    TMat<Scalar> extend_predict_S_t_ = TMat<Scalar>::Zero(2, 1);
    TMat<Scalar> predict_S_t_ = TMat<Scalar>::Zero(1, 1);
    TMat<Scalar> M_ = TMat<Scalar>::Zero(2, 2);
};

/**
 * @brief Static Dimensional Square Root Kalman Filter (SRKF)
 * @tparam StateSize Dimension of the error state vector
 * @tparam ObserveSize Dimension of the measurement vector
 *
 * Algorithm and variables same as SquareRootKalmanFilterDynamic.
 */
template <typename Scalar, int32_t StateSize, int32_t ObserveSize>
class SquareRootKalmanFilterStatic: public Filter<Scalar, SquareRootKalmanFilterStatic<Scalar, StateSize, ObserveSize>> {

    static_assert(StateSize > 0 && ObserveSize > 0, "Size of state and observe must be larger than 0.");

public:
    SquareRootKalmanFilterStatic(): Filter<Scalar, SquareRootKalmanFilterStatic<Scalar, StateSize, ObserveSize>>() {}
    virtual ~SquareRootKalmanFilterStatic() = default;

    bool PropagateCovarianceImpl();
    bool UpdateStateAndCovarianceImpl(const TMat<Scalar> &observation = TMat<Scalar>::Zero(1, 1));

    // Reference for member variables.
    SquareRootKalmanFilterOptions &options() { return options_; }
    TVec<Scalar, StateSize> &dx() { return dx_; }
    TMat<Scalar, StateSize, StateSize> &S_t() { return S_t_; }
    TMat<Scalar, StateSize, StateSize> &predict_S_t() { return predict_S_t_; }
    TMat<Scalar, StateSize, StateSize> &F() { return F_; }
    TMat<Scalar, ObserveSize, StateSize> &H() { return H_; }
    TMat<Scalar, StateSize, StateSize> &sqrt_Q_t() { return sqrt_Q_t_; }
    TMat<Scalar, ObserveSize, ObserveSize> &sqrt_R_t() { return sqrt_R_t_; }

    // Const reference for member variables.
    const SquareRootKalmanFilterOptions &options() const { return options_; }
    const TVec<Scalar, StateSize> &dx() const { return dx_; }
    const TMat<Scalar, StateSize, StateSize> &S_t() const { return S_t_; }
    const TMat<Scalar, StateSize, StateSize> &predict_S_t() const { return predict_S_t_; }
    const TMat<Scalar, StateSize, StateSize> &F() const { return F_; }
    const TMat<Scalar, ObserveSize, StateSize> &H() const { return H_; }
    const TMat<Scalar, StateSize, StateSize> &sqrt_Q_t() const { return sqrt_Q_t_; }
    const TMat<Scalar, ObserveSize, ObserveSize> &sqrt_R_t() const { return sqrt_R_t_; }

private:
    SquareRootKalmanFilterOptions options_;

    TVec<Scalar, StateSize> dx_ = TVec<Scalar, StateSize>::Zero();
    // P is represent as P = S * S.t.
    TMat<Scalar, StateSize, StateSize> S_t_ = TMat<Scalar, StateSize, StateSize>::Zero();

    // Process function F and measurement function H.
    TMat<Scalar, StateSize, StateSize> F_ = TMat<Scalar, StateSize, StateSize>::Identity();
    TMat<Scalar, ObserveSize, StateSize> H_ = TMat<Scalar, ObserveSize, StateSize>::Zero();

    // Process noise Q and measurement noise R.
    // Define Q^(T/2) and R^(T/2) here.
    TMat<Scalar, StateSize, StateSize> sqrt_Q_t_ = TMat<Scalar, StateSize, StateSize>::Zero();
    TMat<Scalar, ObserveSize, ObserveSize> sqrt_R_t_ = TMat<Scalar, ObserveSize, ObserveSize>::Zero();

    TMat<Scalar, StateSize + StateSize, StateSize> extend_predict_S_t_ = TMat<Scalar, StateSize + StateSize, StateSize>::Zero();
    TMat<Scalar, StateSize, StateSize> predict_S_t_ = TMat<Scalar, StateSize, StateSize>::Zero();
    TMat<Scalar, StateSize + ObserveSize, StateSize + ObserveSize> M_ = TMat<Scalar, StateSize + ObserveSize, StateSize + ObserveSize>::Zero();
};

/* Class Square Root Error State Kalman Filter Definition. */
template <typename Scalar, int32_t StateSize, int32_t ObserveSize>
bool SquareRootKalmanFilterStatic<Scalar, StateSize, ObserveSize>::PropagateCovarianceImpl() {
    dx_.setZero();
    /*  extend_predict_S_t_ = [ S.t * F.t ]
                              [   Q.t/2   ] */
    extend_predict_S_t_.template block<StateSize, StateSize>(0, 0) = S_t_ * F_.transpose();
    extend_predict_S_t_.template block<StateSize, StateSize>(StateSize, 0) = sqrt_Q_t_;

    // After QR decomposing of extend_predict_S_t_, the top matrix of the upper triangular matrix becomes predict_S_t_.
    Eigen::HouseholderQR<TMat<Scalar, StateSize + StateSize, StateSize>> qr_solver(extend_predict_S_t_);
    predict_S_t_ = qr_solver.matrixQR().template block<StateSize, StateSize>(0, 0).template triangularView<Eigen::Upper>();
    return true;
}

template <typename Scalar, int32_t StateSize, int32_t ObserveSize>
bool SquareRootKalmanFilterStatic<Scalar, StateSize, ObserveSize>::UpdateStateAndCovarianceImpl(const TMat<Scalar> &residual) {
    // Reference: Optimal State Estimation - Kalman, H infinity, and Nonlinear Approaches.
    // Construct matrix M, do QR decompose on it.
    /*  M = [ R.t/2             0    ] = T * [ (H * pre_P * H.t + R).t/2  hat_K.t ]
            [ pre_S.t * H.t  pre_S.t ]       [             0                S.t   ]
        T is a exist unit orthogonal matrix. */
    M_.template block<ObserveSize, ObserveSize>(0, 0) = sqrt_R_t_;
    M_.template block<ObserveSize, StateSize>(0, ObserveSize).setZero();
    M_.template block<StateSize, ObserveSize>(ObserveSize, 0) = predict_S_t_ * H_.transpose();
    M_.template block<StateSize, StateSize>(ObserveSize, ObserveSize) = predict_S_t_;
    Eigen::HouseholderQR<TMat<Scalar, ObserveSize + StateSize, ObserveSize + StateSize>> qr_solver(M_);
    TMat<Scalar, ObserveSize + StateSize, ObserveSize + StateSize> R_upper = qr_solver.matrixQR().template triangularView<Eigen::Upper>();

    // Compute Kalman gain.
    // hat_K = (H * pre_P * H.t + R).t/2 * K.
    const TMat<Scalar, ObserveSize, ObserveSize> sqrt_S_t = R_upper.template block<ObserveSize, ObserveSize>(0, 0);
    const TMat<Scalar, ObserveSize, StateSize> hat_K_t = R_upper.template block<ObserveSize, StateSize>(0, ObserveSize);

    // K = hat_K * (sqrt_S_t)^-1 -> K.t = (sqrt_S_t.t)^-1 * hat_K.t
    const TMat<Scalar, StateSize, ObserveSize> K_ = hat_K_t.transpose() * sqrt_S_t.template triangularView<Eigen::Upper>().solve(TMat<Scalar, ObserveSize, ObserveSize>::Identity());

    // Update error state.
    dx_ = K_ * residual;

    // Update covariance of new state.
    switch (options_.kMethod) {
        default:
        case StateCovUpdateMethod::kSimple:
        case StateCovUpdateMethod::kFull:
            S_t_ = R_upper.template block<StateSize, StateSize>(ObserveSize, ObserveSize);
            break;
    }

    return true;
}

}  // namespace slam_solver

#endif  // end of _SQUARE_ROOT_KALMAN_FILTER_SOLVER_H_
