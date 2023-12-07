#ifndef _GENERAL_GRAPH_OPTIMIZOR_KERNEL_HUBER_H_
#define _GENERAL_GRAPH_OPTIMIZOR_KERNEL_HUBER_H_

#include "kernel.h"

namespace SLAM_SOLVER {

/* Class Kernel Huber Declaration. */
template <typename Scalar>
class KernelHuber : public Kernel<Scalar> {

public:
    KernelHuber() = default;
    explicit KernelHuber(Scalar delta) : delta_(delta), dsqr_(delta * delta) {}
    virtual ~KernelHuber() = default;

    virtual void Compute(const Scalar x) override {
        this->x() = x;

        if (this->x() > dsqr_) {
            const Scalar sqrt_x = std::sqrt(this->x());
            this->y(0) = 2.0f * sqrt_x * delta_ - dsqr_;
            this->y(1) = delta_ / sqrt_x;
        } else {
            this->y(0) = this->x();
            this->y(1) = static_cast<Scalar>(1);
        }
    }

private:
    Scalar delta_ = static_cast<Scalar>(1);
    Scalar dsqr_ = static_cast<Scalar>(1);

};

}

#endif // end of _GENERAL_GRAPH_OPTIMIZOR_KERNEL_HUBER_H_
