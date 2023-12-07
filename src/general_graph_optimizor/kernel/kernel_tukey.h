#ifndef _GENERAL_GRAPH_OPTIMIZOR_KERNEL_TUKEY_H_
#define _GENERAL_GRAPH_OPTIMIZOR_KERNEL_TUKEY_H_

#include "kernel.h"

namespace SLAM_SOLVER {

/* Class Kernel Cauchy Declaration. */
template <typename Scalar>
class KernelTukey : public Kernel<Scalar> {

public:
    KernelTukey() = default;
    explicit KernelTukey(Scalar delta) : delta_(delta), dsqr_(delta * delta) {}
    virtual ~KernelTukey() = default;

    virtual void Compute(const Scalar x) override {
        this->x() = x;

        const Scalar sqrt_x = std::sqrt(x);
        if (sqrt_x < this->delta_) {
            const Scalar aux = x / this->delta_;
            const Scalar _aux = static_cast<Scalar>(1) - aux;
            const Scalar _aux_2 = _aux * _aux;
            const Scalar _aux_3 = _aux_2 * _aux;

            this->y(0) = this->dsqr_ * (static_cast<Scalar>(1) - _aux_3) / static_cast<Scalar>(3);
            this->y(1) = _aux_2;
        } else {
            this->y(0) = this->dsqr_ / static_cast<Scalar>(3);
            this->y(1) = static_cast<Scalar>(0);
        }
    }

private:
    Scalar delta_ = static_cast<Scalar>(1);
    Scalar dsqr_ = static_cast<Scalar>(1);

};

}

#endif // end of _GENERAL_GRAPH_OPTIMIZOR_KERNEL_TUKEY_H_
