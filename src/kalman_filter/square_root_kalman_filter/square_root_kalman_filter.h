#ifndef _SQUARE_ROOT_KALMAN_FILTER_SOLVER_H_
#define _SQUARE_ROOT_KALMAN_FILTER_SOLVER_H_

#include "datatype_basic.h"
#include "filter.h"

namespace SLAM_SOLVER {

struct SquareRootKalmanFilterOptions {
    StateCovUpdateMethod kMethod = StateCovUpdateMethod::kSimple;
};

/* Class Square Root Error State Kalman Filter Declaration. */
template <typename Scalar>
class SquareRootKalmanFilterDynamic : public Filter<Scalar, SquareRootKalmanFilterDynamic<Scalar>> {

public:
    SquareRootKalmanFilterDynamic() : Filter<Scalar, SquareRootKalmanFilterDynamic<Scalar>>() {}
    virtual ~SquareRootKalmanFilterDynamic() = default;

    bool PropagateNominalStateImpl(const TVec<Scalar> &parameters = TVec1<Scalar>());
    bool PropagateCovarianceImpl();
    bool UpdateStateAndCovarianceImpl(const TMat<Scalar> &observation = TVec1<Scalar>());

    // Reference for member variables.
    SquareRootKalmanFilterOptions &options() { return options_; }
    TVec<Scalar> &dx() { return dx_; }
    TMat<Scalar> &S_t() { return S_t_; }
    TMat<Scalar> &F() { return F_; }
    TMat<Scalar> &H() { return H_; }
    TMat<Scalar> &square_Q_t() { return square_Q_t_; }
    TMat<Scalar> &square_R_t() { return square_R_t_; }

    // Const reference for member variables.
    const SquareRootKalmanFilterOptions &options() const { return options_; }
    const TVec<Scalar> &dx() const { return dx_; }
    const TMat<Scalar> &S_t() const { return S_t_; }
    const TMat<Scalar> &F() const { return F_; }
    const TMat<Scalar> &H() const { return H_; }
    const TMat<Scalar> &square_Q_t() const { return square_Q_t_; }
    const TMat<Scalar> &square_R_t() const { return square_R_t_; }

private:
    SquareRootKalmanFilterOptions options_;

    TVec<Scalar> dx_ = TVec1<Scalar>::Zero();
    // P is represent as P = S * S.t.
    TMat<Scalar> S_t_ = TMat1<Scalar>::Zero();

    // Process function F and measurement function H.
    TMat<Scalar> F_ = TMat1<Scalar>::Identity();
    TMat<Scalar> H_ = TMat1<Scalar>::Identity();

    // Process noise Q and measurement noise R.
    // Define Q^(T/2) and R^(T/2) here.
    TMat<Scalar> square_Q_t_ = TMat1<Scalar>::Zero();
    TMat<Scalar> square_R_t_ = TMat1<Scalar>::Zero();

    TMat<Scalar> extend_predict_S_t_ = TVec2<Scalar>::Zero();
    TMat<Scalar> predict_S_t_ = TMat1<Scalar>::Zero();
    TMat<Scalar> M_ = TMat2<Scalar>::Zero();

};

/* Class Square Root Error State Kalman Filter Declaration. */
template <typename Scalar, int32_t StateSize, int32_t ObserveSize>
class SquareRootKalmanFilterStatic : public Filter<Scalar, SquareRootKalmanFilterStatic<Scalar, StateSize, ObserveSize>> {

static_assert(StateSize > 0 && ObserveSize > 0, "Size of state and observe must be larger than 0.");

public:
    SquareRootKalmanFilterStatic() : Filter<Scalar, SquareRootKalmanFilterStatic<Scalar, StateSize, ObserveSize>>() {}
    virtual ~SquareRootKalmanFilterStatic() = default;

    bool PropagateNominalStateImpl(const TVec<Scalar> &parameters = TVec<Scalar, 1>());
    bool PropagateCovarianceImpl();
    bool UpdateStateAndCovarianceImpl(const TMat<Scalar> &observation = TVec<Scalar, 1>());

    // Reference for member variables.
    SquareRootKalmanFilterOptions &options() { return options_; }
    TVec<Scalar, StateSize> &dx() { return dx_; }
    TMat<Scalar, StateSize, StateSize> &S_t() { return S_t_; }
    TMat<Scalar, StateSize, StateSize> &F() { return F_; }
    TMat<Scalar, ObserveSize, StateSize> &H() { return H_; }
    TMat<Scalar, StateSize, StateSize> &square_Q_t() { return square_Q_t_; }
    TMat<Scalar, ObserveSize, ObserveSize> &square_R_t() { return square_R_t_; }

    // Const reference for member variables.
    const SquareRootKalmanFilterOptions &options() const { return options_; }
    const TVec<Scalar, StateSize> &dx() const { return dx_; }
    const TMat<Scalar, StateSize, StateSize> &S_t() const { return S_t_; }
    const TMat<Scalar, StateSize, StateSize> &F() const { return F_; }
    const TMat<Scalar, ObserveSize, StateSize> &H() const { return H_; }
    const TMat<Scalar, StateSize, StateSize> &square_Q_t() const { return square_Q_t_; }
    const TMat<Scalar, ObserveSize, ObserveSize> &square_R_t() const { return square_R_t_; }

private:
    SquareRootKalmanFilterOptions options_;

    TVec<Scalar, StateSize> dx_ = TVec<Scalar, StateSize>::Zero();
    // P is represent as P = S * S.t.
    TMat<Scalar, StateSize, StateSize> S_t_ = TMat<Scalar, StateSize, StateSize>::Zero();

    // Process function F and measurement function H.
    TMat<Scalar, StateSize, StateSize> F_ = TMat<Scalar, StateSize, StateSize>::Identity();
    TMat<Scalar, ObserveSize, StateSize> H_ = TMat<Scalar, ObserveSize, StateSize>::Identity();

    // Process noise Q and measurement noise R.
    // Define Q^(T/2) and R^(T/2) here.
    TMat<Scalar, StateSize, StateSize> square_Q_t_ = TMat<Scalar, StateSize, StateSize>::Zero();
    TMat<Scalar, ObserveSize, ObserveSize> square_R_t_ = TMat<Scalar, ObserveSize, ObserveSize>::Zero();

    TMat<Scalar, StateSize + StateSize, StateSize> extend_predict_S_t_ = TMat<Scalar, StateSize + StateSize, StateSize>::Zero();
    TMat<Scalar, StateSize, StateSize> predict_S_t_ = TMat<Scalar, StateSize, StateSize>::Zero();
    TMat<Scalar, StateSize + ObserveSize, StateSize + ObserveSize> M_ = TMat<Scalar, StateSize + ObserveSize, StateSize + ObserveSize>::Zero();

};

/* Class Square Root Error State Kalman Filter Definition. */
template <typename Scalar, int32_t StateSize, int32_t ObserveSize>
bool SquareRootKalmanFilterStatic<Scalar, StateSize, ObserveSize>::PropagateNominalStateImpl(const TVec<Scalar> &parameters) {
    return true;
}

template <typename Scalar, int32_t StateSize, int32_t ObserveSize>
bool SquareRootKalmanFilterStatic<Scalar, StateSize, ObserveSize>::PropagateCovarianceImpl() {
    /*  extend_predict_S_t_ = [ S.t * F.t ]
                              [   Q.t/2   ] */
    extend_predict_S_t_.template block<StateSize, StateSize>(0, 0) = S_t_ * F_.transpose();
    extend_predict_S_t_.template block<StateSize, StateSize>(StateSize, 0) = square_Q_t_;

    // After QR decomposing of extend_predict_S_t_, the top matrix of the upper triangular matrix becomes predict_S_t_.
    Eigen::HouseholderQR<TMat<Scalar, StateSize + StateSize, StateSize>> qr_solver(extend_predict_S_t_);
    extend_predict_S_t_ = qr_solver.matrixQR().template triangularView<Eigen::Upper>();
    predict_S_t_ = extend_predict_S_t_.template block<StateSize, StateSize>(0, 0);
    return true;
}

template <typename Scalar, int32_t StateSize, int32_t ObserveSize>
bool SquareRootKalmanFilterStatic<Scalar, StateSize, ObserveSize>::UpdateStateAndCovarianceImpl(const TMat<Scalar> &residual) {
    // Construct matrix M, do QR decompose on it.
    /*  M = [ R.t/2             0    ] = T * [ (H * pre_P * H.t + R).t/2  hat_K.t ]
            [ pre_S.t * H.t  pre_S.t ]       [             0                S.t   ]
        T is a exist unit orthogonal matrix. */
    M_.template block<ObserveSize, ObserveSize>(0, 0) = square_R_t_;
    M_.template block<ObserveSize, StateSize>(0, ObserveSize).setZero();
    M_.template block<StateSize, ObserveSize>(ObserveSize, 0) = predict_S_t_ * H_.transpose();
    M_.template block<StateSize, StateSize>(ObserveSize, ObserveSize) = predict_S_t_;
    Eigen::HouseholderQR<TMat<Scalar, ObserveSize + StateSize, ObserveSize + StateSize>> qr_solver(M_);
    M_ = qr_solver.matrixQR().template triangularView<Eigen::Upper>();

    // Commpute Kalman gain.
    // hat_K = (H * pre_P * H.t + R).t/2 * K.
    const TMat<Scalar, StateSize, ObserveSize> K_
        = M_.template block<ObserveSize, StateSize>(0, ObserveSize).transpose()
        * M_.template block<ObserveSize, ObserveSize>(0, 0).inverse();

    // Update error state.
    dx_ = K_ * residual;

    // Update covariance of new state.
    switch (options_.kMethod) {
        default:
        case StateCovUpdateMethod::kSimple:
        case StateCovUpdateMethod::kFull:
            S_t_ = M_.template block<StateSize, StateSize>(ObserveSize, ObserveSize);
            break;
    }

    return true;
}

}

#endif // end of _SQUARE_ROOT_KALMAN_FILTER_SOLVER_H_
