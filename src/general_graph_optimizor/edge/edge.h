#ifndef _GENERAL_GRAPH_OPTIMIZOR_EDGE_H_
#define _GENERAL_GRAPH_OPTIMIZOR_EDGE_H_

#include "vector"
#include "memory"

#include "datatype_basic.h"
#include "vertex.h"
#include "kernel.h"

namespace SLAM_SOLVER {

/* Class Edge Declaration. */
template <typename Scalar>
class Edge {

public:
    Edge() = delete;
    Edge(int32_t residual_dim, int32_t vertex_num);
    virtual ~Edge() = default;

    // Edge index.
    static uint32_t &GetGlobalId() { return global_id_; }
    const uint32_t GetId() const { return id_; }

    // Use string to represent edge type.
    virtual std::string GetType();

private:
    // Global index for every vertex.
    static uint32_t global_id_;
    uint32_t id_ = 0;

    // Kernel function, nullptr default.
    std::unique_ptr<Kernel<Scalar>> kernel_ = nullptr;

    // Residual for this edge, jacobians for all vertice.
    TVec<Scalar> residual_ = TVec3<Scalar>::Zero();
    std::vector<TMat<Scalar>> jacobians_ = {};
};

/* Class Edge Definition. */
template <typename Scalar>
uint32_t Edge<Scalar>::global_id_ = 0;

template <typename Scalar>
Edge<Scalar>::Edge(int32_t residual_dim, int32_t vertex_num) {
    // Resize size of residual and jacobian.
    if (residual_.size() != residual_dim) {
        residual_.resize(residual_dim);
    }
    jacobians_.resize(vertex_num);

    // Set index.
    ++Edge<Scalar>::global_id_;
    id_ = Edge<Scalar>::global_id_;
}

template <typename Scalar>
std::string Edge<Scalar>::GetType() {
    return std::string("Basic Edge");
}

}

#endif // end of _GENERAL_GRAPH_OPTIMIZOR_EDGE_H_
