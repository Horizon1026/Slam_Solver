#ifndef _KALMAN_FILTER_SOLVER_
#define _KALMAN_FILTER_SOLVER_

#include "datatype_basic.h"
#include "filter.h"

namespace SLAM_SOLVER {

/* Class Basic Kalman Filter Declaration. */
template <typename Scalar, int32_t StateSize = -1, int32_t ObserveSize = -1>
class KalmanFilter : public Filter<Scalar, KalmanFilter<Scalar, StateSize, ObserveSize>> {

public:
    KalmanFilter() : Filter<Scalar, KalmanFilter<Scalar, StateSize, ObserveSize>>() {}
    virtual ~KalmanFilter() = default;

    bool Propagate(const TVec<Scalar> &parameters = TVec<Scalar, 1>());
    bool Update(const TMat<Scalar> &observation = TVec<Scalar, 1>());

    TVec<Scalar, StateSize> &x() { return x_; }
    TMat<Scalar, StateSize, StateSize> &P() { return P_; }
    TMat<Scalar, StateSize, StateSize> &F() { return F_; }
    TMat<Scalar, ObserveSize, StateSize> &H() { return H_; }
    TMat<Scalar, StateSize, StateSize> &Q() { return Q_; }
    TMat<Scalar, ObserveSize, ObserveSize> &R() { return R_; }

private:
    TVec<Scalar, StateSize> x_ = TVec<Scalar, StateSize>::Zero();
    TMat<Scalar, StateSize, StateSize> P_ = TMat<Scalar, StateSize, StateSize>::Zero();

    TVec<Scalar, StateSize> predict_x_ = TVec<Scalar, StateSize>::Zero();
    TMat<Scalar, StateSize, StateSize> predict_P_ = TMat<Scalar, StateSize, StateSize>::Zero();
    TMat<Scalar, ObserveSize, ObserveSize> predict_S_ = TMat<Scalar, ObserveSize, ObserveSize>::Zero();

    // Process function F and measurement function H.
    TMat<Scalar, StateSize, StateSize> F_ = TMat<Scalar, StateSize, StateSize>::Identity();
    TMat<Scalar, ObserveSize, StateSize> H_ = TMat<Scalar, ObserveSize, StateSize>::Identity();

    // Process noise Q and measurement noise R.
    TMat<Scalar, StateSize, StateSize> Q_ = TMat<Scalar, StateSize, StateSize>::Zero();
    TMat<Scalar, ObserveSize, ObserveSize> R_ = TMat<Scalar, ObserveSize, ObserveSize>::Zero();

};

/* Class Basic Kalman Filter Definition. */
template <typename Scalar, int32_t StateSize, int32_t ObserveSize>
bool KalmanFilter<Scalar, StateSize, ObserveSize>::Propagate(const TVec<Scalar> &parameters) {
    predict_x_ = F_ * x_;
    predict_P_ = F_ * P_ * F_.transpose() + Q_;
    return true;
}

template <typename Scalar, int32_t StateSize, int32_t ObserveSize>
bool KalmanFilter<Scalar, StateSize, ObserveSize>::Update(const TMat<Scalar> &observation) {
    const TMat<Scalar, ObserveSize, StateSize> H_t = H_.transpose();

    // Compute Kalman gain.
    predict_S_ = H_ * predict_P_ * H_t + R_;
    const TMat<Scalar, StateSize, ObserveSize> K_ = predict_P_ * H_t * predict_S_.inverse();

    // Compute new information.
    const TVec<Scalar, StateSize> v_ = observation - H_ * predict_x_;

    // Update new state.
    x_ = predict_x_ + K_ * v_;

    // Update covariance of new state.
    TMat<Scalar, StateSize, StateSize> I_KH = -K_ * H_;
    // for (int32_t i = 0; i < StateSize; ++i) {
    //     I_KH(i, i) +=
    // }
    I_KH.diagonal().array() += static_cast<Scalar>(1);
    P_ = I_KH * predict_P_;

    return true;
}

};

#endif // end of _KALMAN_FILTER_SOLVER_
