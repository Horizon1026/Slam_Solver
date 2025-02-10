#ifndef _SQUARE_ROOT_INFORMATION_FILTER_SOLVER_H_
#define _SQUARE_ROOT_INFORMATION_FILTER_SOLVER_H_

#include "basic_type.h"
#include "inverse_filter.h"

namespace SLAM_SOLVER {

/* Class Square Root Error State Information Filter Declaration. */
template <typename Scalar>
class SquareRootInformationFilterDynamic : public InverseFilter<Scalar, SquareRootInformationFilterDynamic<Scalar>> {

public:
    SquareRootInformationFilterDynamic() : InverseFilter<Scalar, SquareRootInformationFilterDynamic<Scalar>>() {}
    virtual ~SquareRootInformationFilterDynamic() = default;

    bool PropagateNominalStateImpl(const TVec<Scalar> &parameters = TMat<Scalar>::Zero(1, 1));
    bool PropagateInformationImpl();
    bool UpdateStateAndInformationImpl(const TMat<Scalar> &observation = TMat<Scalar>::Zero(1, 1));

    // Reference for member variables.
    TVec<Scalar> &dx() { return dx_; }
    TMat<Scalar> &kesi_t() { return kesi_t_; }
    TMat<Scalar> &F() { return F_; }
    TMat<Scalar> &H() { return H_; }
    TMat<Scalar> &inv_sqrt_Q_t() { return inv_sqrt_Q_t_; }
    TMat<Scalar> &inv_sqrt_R_t() { return inv_sqrt_R_t_; }
    TMat<Scalar> &predict_kesi_t() { return predict_kesi_t_; }

    // Const reference for member variables.
    const TVec<Scalar> &dx() const { return dx_; }
    const TMat<Scalar> &kesi_t() const { return kesi_t_; }
    const TMat<Scalar> &F() const { return F_; }
    const TMat<Scalar> &H() const { return H_; }
    const TMat<Scalar> &inv_sqrt_Q_t() const { return inv_sqrt_Q_t_; }
    const TMat<Scalar> &inv_sqrt_R_t() const { return inv_sqrt_R_t_; }
    const TMat<Scalar> &predict_kesi_t() const { return predict_kesi_t_; }

private:
    TVec<Scalar> dx_ = TMat<Scalar>::Zero(1, 1);
    // I is represent as kesi * kesi.t.
    TMat<Scalar> kesi_t_ = TMat<Scalar>::Zero(1, 1);

    // Process function F and measurement function H.
    TMat<Scalar> F_ = TMat<Scalar>::Identity(1, 1);
    TMat<Scalar> H_ = TMat<Scalar>::Identity(1, 1);

    // Process noise Q and measurement noise R.
    // Define Q^(-T/2) and R^(-T/2) here.
    TMat<Scalar> inv_sqrt_Q_t_ = TMat<Scalar>::Zero(1, 1);
    TMat<Scalar> inv_sqrt_R_t_ = TMat<Scalar>::Zero(1, 1);

    TMat<Scalar> extend_rho_t_ = TMat<Scalar>::Zero(2, 1);
    TMat<Scalar> extend_predict_kesi_t_ = TMat<Scalar>::Zero(2, 1);
    TMat<Scalar> predict_kesi_t_ = TMat<Scalar>::Zero(1, 1);
    TMat<Scalar> M_ = TMat<Scalar>::Zero(2, 1);

};

}

#endif // end of _SQUARE_ROOT_INFORMATION_FILTER_SOLVER_H_
