#ifndef _GENERAL_GRAPH_OPTIMIZOR_KERNEL_CAUCHY_H_
#define _GENERAL_GRAPH_OPTIMIZOR_KERNEL_CAUCHY_H_

#include "kernel.h"

namespace slam_solver {

/* Class Kernel Cauchy Declaration. */
template <typename Scalar>
class KernelCauchy : public Kernel<Scalar> {

public:
    KernelCauchy() = default;
    explicit KernelCauchy(Scalar delta)
        : delta_(delta)
        , dsqr_(delta * delta)
        , inv_dsqr_(static_cast<Scalar>(1) / delta / delta) {}
    virtual ~KernelCauchy() = default;

    virtual void Compute(const Scalar x) override {
        this->x() = x;
        const Scalar tmp = this->inv_dsqr_ * x + static_cast<Scalar>(1);
        this->y(0) = this->dsqr_ * std::log(tmp);
        this->y(1) = static_cast<Scalar>(1) / tmp;
    }

private:
    Scalar delta_ = static_cast<Scalar>(1);
    Scalar dsqr_ = static_cast<Scalar>(1);
    Scalar inv_dsqr_ = static_cast<Scalar>(1);
};

}  // namespace slam_solver

#endif  // end of _GENERAL_GRAPH_OPTIMIZOR_KERNEL_CAUCHY_H_
