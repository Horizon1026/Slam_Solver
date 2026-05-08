#ifndef _GENERAL_GRAPH_OPTIMIZOR_VERTEX_QUATERNION_H_
#define _GENERAL_GRAPH_OPTIMIZOR_VERTEX_QUATERNION_H_

#include "basic_type.h"
#include "vertex.h"

namespace slam_solver {

/* Class Vertex Quaternion declaration. */
template <typename Scalar>
class VertexQuat: public Vertex<Scalar> {

public:
    VertexQuat(): Vertex<Scalar>(4, 3) {}
    virtual ~VertexQuat() = default;

    // Update param with delta_param solved by solver.
    virtual void UpdateParam(const TVec<Scalar> &delta_param) override {
        // Stored param is [w, x, y, z]. Incremental param is [dx, dy, dz].
        TQuat<Scalar> q(this->param()(0), this->param()(1), this->param()(2), this->param()(3));
        TQuat<Scalar> dq(1, 0.5 * delta_param(0), 0.5 * delta_param(1), 0.5 * delta_param(2));
        q = q * dq;
        q.normalize();
        this->param() << q.w(), q.x(), q.y(), q.z();
    }

    // Compute manifold jacobian dx/d_delta.
    // Mathematical Principle: Manifold derivative of unit quaternion (4D) w.r.t. 3D Lie algebra increment
    // Jacobian = 0.5 * [ -x  -y  -z ]
    //                  [  w  -z   y ]
    //                  [  z   w  -x ]
    //                  [ -y   x   w ]
    // Derived from quaternion right-increment update rule: q <- q * Exp(delta)
    // This matrix maps 3D tangent space delta to 4D quaternion parameter space derivative
    virtual void ComputeManifoldJacobian(TMat<Scalar> &jacobian) const override {
        jacobian.setZero(4, 3);
        const Scalar w = this->param()(0);
        const Scalar x = this->param()(1);
        const Scalar y = this->param()(2);
        const Scalar z = this->param()(3);
        jacobian << -x, -y, -z,
                     w, -z,  y,
                     z,  w, -x,
                    -y,  x,  w;
        jacobian *= static_cast<Scalar>(0.5);
    }
};

}  // namespace slam_solver

#endif  // end of _GENERAL_GRAPH_OPTIMIZOR_VERTEX_QUATERNION_H_
