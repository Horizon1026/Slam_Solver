#ifndef _GENERAL_GRAPH_OPTIMIZOR_KERNEL_TRIVAL_H_
#define _GENERAL_GRAPH_OPTIMIZOR_KERNEL_TRIVAL_H_

#include "kernel.h"

namespace SLAM_SOLVER {

/* Class Kernel Trival Declaration. */
template <typename Scalar>
class KernelTrival : public Kernel<Scalar> {

public:
    KernelTrival() = default;
    virtual ~KernelTrival() = default;

    virtual void Compute(const Scalar x) override {
        x_ = x;
        y_[0] = x_;
        y_[1] = static_cast<Scalar>(1);
        y_[2] = static_cast<Scalar>(0);
    }

};

}

#endif // end of _GENERAL_GRAPH_OPTIMIZOR_KERNEL_TRIVAL_H_
