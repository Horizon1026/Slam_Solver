#ifndef _SLAM_FILTER_H_
#define _SLAM_FILTER_H_

#include "basic_type.h"

#include "memory"
#include "type_traits"

namespace slam_solver {

enum StateCovUpdateMethod : uint8_t {
    kSimple = 0,
    kFull = 1,
};

/* Template Class Filter Declaration. */
template <typename Scalar, typename FilterType>
class Filter {

public:
    Filter() = default;
    virtual ~Filter() = default;
    Filter(const Filter &filter) = delete;

    bool PropagateCovariance() { return reinterpret_cast<FilterType *>(this)->PropagateCovarianceImpl(); }

    bool UpdateStateAndCovariance(const TMat<Scalar> &value = TMat<Scalar>::Zero(1, 1)) {
        return reinterpret_cast<FilterType *>(this)->UpdateStateAndCovarianceImpl(value);
    }

    // Remove the component of dx along the supplied directions.
    template <typename DirectionsDerived, typename IncrementDerived>
    static void RemoveStateIncrementAlongDirections(const Eigen::MatrixBase<DirectionsDerived> &directions,
                                                   Eigen::MatrixBase<IncrementDerived> &dx) {
        static_assert(std::is_same_v<Scalar, typename IncrementDerived::Scalar>, "Increment must use the filter scalar");
        static_assert(std::is_floating_point_v<Scalar>, "State increments must use a real floating-point scalar");
        static_assert(std::is_same_v<Scalar, typename DirectionsDerived::Scalar>, "Directions and increment must use the same scalar");
        eigen_assert(dx.cols() == 1);
        if (directions.cols() == 0) {
            return;
        }
        eigen_assert(directions.rows() == dx.rows());
        Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> basis = directions;
        for (Eigen::Index col = 0; col < basis.cols(); ++col) {
            const Scalar norm = basis.col(col).stableNorm();
            if (norm > Scalar(0)) {
                basis.col(col) /= norm;
            }
        }
        const Eigen::ColPivHouseholderQR<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>> qr(basis);
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1> rotated_dx = qr.householderQ().adjoint() * dx;
        rotated_dx.head(qr.rank()).setZero();
        dx.noalias() = qr.householderQ() * rotated_dx;
    }
};

}  // namespace slam_solver

#endif  // end of _SLAM_FILTER_H_
