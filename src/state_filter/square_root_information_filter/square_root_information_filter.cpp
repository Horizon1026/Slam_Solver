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
    // TODO:
    return true;
}

template <typename Scalar>
bool SquareRootInformationFilterDynamic<Scalar>::UpdateStateAndInformationImpl(const TMat<Scalar> &residual) {
    // TODO:
    return true;
}

}
