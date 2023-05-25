#ifndef _SQUARE_ROOT_KALMAN_FILTER_SOLVER_H_
#define _SQUARE_ROOT_KALMAN_FILTER_SOLVER_H_

#include "datatype_basic.h"
#include "filter.h"

namespace SLAM_SOLVER {

struct SquareRootKalmanFilterOptions {
    StateCovUpdateMethod kMethod = StateCovUpdateMethod::kSimple;
};

/* Class Basic Kalman Filter Declaration. */
template <typename Scalar, int32_t StateSize = -1, int32_t ObserveSize = -1>
class SquareRootKalmanFilter : public Filter<Scalar, SquareRootKalmanFilter<Scalar, StateSize, ObserveSize>> {

public:
    SquareRootKalmanFilter() : Filter<Scalar, SquareRootKalmanFilter<Scalar, StateSize, ObserveSize>>() {}
    virtual ~SquareRootKalmanFilter() = default;

    bool PropagateNominalStateImpl(const TVec<Scalar> &parameters = TVec<Scalar, 1>());
    bool PropagateCovarianceImpl(const TVec<Scalar> &parameters = TVec<Scalar, 1>());
    bool UpdateStateAndCovarianceImpl(const TMat<Scalar> &observation = TVec<Scalar, 1>());

    SquareRootKalmanFilterOptions &options() { return options_; }

private:
    SquareRootKalmanFilterOptions options_;

    TVec<Scalar, StateSize> dx_ = TVec<Scalar, StateSize>::Zero();
    TMat<Scalar, StateSize, StateSize> square_P_ = TMat<Scalar, StateSize, StateSize>::Zero();

    TMat<Scalar, StateSize * 2, StateSize> extend_predict_square_P_ = TMat<Scalar, StateSize * 2, StateSize>::Zero();
    TMat<Scalar, StateSize, StateSize> predict_square_P_ = TMat<Scalar, StateSize, StateSize>::Zero();
    TMat<Scalar, StateSize + ObserveSize, StateSize + ObserveSize> M_ = TMat<Scalar, StateSize + ObserveSize, StateSize + ObserveSize>::Zero();

    // Process function F and measurement function H.
    TMat<Scalar, StateSize, StateSize> F_ = TMat<Scalar, StateSize, StateSize>::Identity();
    TMat<Scalar, ObserveSize, StateSize> H_ = TMat<Scalar, ObserveSize, StateSize>::Identity();

    // Process noise Q and measurement noise R.
    // Define Q^(T/2) and R^(T/2) here.
    TMat<Scalar, StateSize, StateSize> square_transpose_Q_ = TMat<Scalar, StateSize, StateSize>::Zero();
    TMat<Scalar, ObserveSize, ObserveSize> square_transpose_R_ = TMat<Scalar, ObserveSize, ObserveSize>::Zero();

};

/* Class Basic Kalman Filter Definition. */
template <typename Scalar, int32_t StateSize, int32_t ObserveSize>
bool SquareRootKalmanFilter<Scalar, StateSize, ObserveSize>::PropagateNominalStateImpl(const TVec<Scalar> &parameters) {
    return true;
}

template <typename Scalar, int32_t StateSize, int32_t ObserveSize>
bool SquareRootKalmanFilter<Scalar, StateSize, ObserveSize>::PropagateCovarianceImpl(const TVec<Scalar> &parameters) {
    // P is represent as P = S * S.t.
    extend_predict_square_P_.block<StateSize, StateSize>(0, 0) = (F_ * square_P_).transpose();
    extend_predict_square_P_.block<StateSize, StateSize>(StateSize, 0) = square_transpose_Q_;
    Eigen::HouseholderQR<TMat<Scalar, StateSize * 2, StateSize>> qr_solver(extend_predict_square_P_);
    predict_square_P_ = qr_solver.matrixQR().template triangularView<Eigen::Upper>();
    return true;
}

template <typename Scalar, int32_t StateSize, int32_t ObserveSize>
bool SquareRootKalmanFilter<Scalar, StateSize, ObserveSize>::UpdateStateAndCovarianceImpl(const TMat<Scalar> &residual) {
    const TMat<Scalar, ObserveSize, StateSize> H_t = H_.transpose();

    // Compute Kalman gain.

    // Update error state.

    // Update covariance of new state.
    switch (options_.kMethod) {
        default:
        case StateCovUpdateMethod::kSimple:
            break;
        case StateCovUpdateMethod::kFull:
            break;
    }

    return true;
}

}

#endif // end of _SQUARE_ROOT_KALMAN_FILTER_SOLVER_H_
