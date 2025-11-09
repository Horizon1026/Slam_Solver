#ifndef _GENERAL_GRAPH_OPTIMIZOR_VERTEX_SO3_H_
#define _GENERAL_GRAPH_OPTIMIZOR_VERTEX_SO3_H_

#include "basic_type.h"
#include "slam_basic_math.h"
#include "vertex.h"

namespace slam_solver {

/* Class Vertex SO3 declaration. */
template <typename Scalar>
class VertexSO3: public Vertex<Scalar> {

public:
    VertexSO3(): Vertex<Scalar>(4, 3) {}
    virtual ~VertexSO3() = default;

    // Update param with delta_param solved by solver.
    virtual void UpdateParam(const TVec<Scalar> &delta_param) override {
        // Stored param is [w, x, y, z]. Incremental param is [dx, dy, dz].
        TQuat<Scalar> q(this->param()(0), this->param()(1), this->param()(2), this->param()(3));
        const TQuat<Scalar> dq = Utility::Exponent(TVec3<Scalar>(delta_param));
        q = q * dq;
        q.normalize();
        this->param() << q.w(), q.x(), q.y(), q.z();
    }
};

}  // namespace slam_solver

#endif  // end of _GENERAL_GRAPH_OPTIMIZOR_VERTEX_SO3_H_
