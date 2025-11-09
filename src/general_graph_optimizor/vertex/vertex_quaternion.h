#ifndef _GENERAL_GRAPH_OPTIMIZOR_VERTEX_QUATERNION_H_
#define _GENERAL_GRAPH_OPTIMIZOR_VERTEX_QUATERNION_H_

#include "basic_type.h"
#include "vertex.h"

namespace SLAM_SOLVER {

/* Class Vertex Quaternion declaration. */
template <typename Scalar>
class VertexQuat : public Vertex<Scalar> {

public:
    VertexQuat()
        : Vertex<Scalar>(4, 3) {}
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
};

}  // namespace SLAM_SOLVER

#endif  // end of _GENERAL_GRAPH_OPTIMIZOR_VERTEX_QUATERNION_H_
