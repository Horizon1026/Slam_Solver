#ifndef _GENERAL_GRAPH_OPTIMIZOR_KERNEL_H_
#define _GENERAL_GRAPH_OPTIMIZOR_KERNEL_H_

#include "datatype_basic.h"

namespace SLAM_SOLVER {

/* Class Kernel Declaration. */
template <typename Scalar>
class Kernel {

public:
    Kernel() = default;
    virtual ~Kernel() = default;

    virtual void Compute(const Scalar x) {
        x_ = x;
        y_[0] = x_;
        y_[1] = static_cast<Scalar>(1);
        y_[2] = static_cast<Scalar>(0);
    }

    const Scalar &x() const { return x_; }
    const Scalar &y(int32_t index) const { return y_[index]; }

private:
    Scalar x_ = 0;
    std::array<Scalar, 3> y_ = {};

};

}

#endif // end of _GENERAL_GRAPH_OPTIMIZOR_KERNEL_H_
