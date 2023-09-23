#ifndef _GENERAL_GRAPH_OPTIMIZOR_EDGE_H_
#define _GENERAL_GRAPH_OPTIMIZOR_EDGE_H_

#include "datatype_basic.h"
#include "log_report.h"
#include "vertex.h"
#include "kernel.h"

#include "vector"
#include "memory"

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

    // Set and get vertex of this edge.
    bool SetVertex(Vertex<Scalar> * vertex, uint32_t index);
    bool SetVertices(const std::vector<Vertex<Scalar> *> &vertices);
    Vertex<Scalar> *GetVertex(uint32_t index) { return vertices_[index]; }
    const std::vector<Vertex<Scalar> *> &GetVertices() const { return vertices_; }
    uint32_t GetVertexNum() const { return vertices_.size(); }

    // Set and get jacobians of each vertex of this edge.
    TMat<Scalar> &GetJacobian(uint32_t index) { return jacobians_[index]; }
    const std::vector<TMat<Scalar>> &GetJacobians() const { return jacobians_; }

    // Reference for member variables.
    TVec<Scalar> &residual() { return residual_; }
    TMat<Scalar> &information() { return information_; }
    TMat<Scalar> &observation() { return observation_; }
    std::unique_ptr<Kernel<Scalar>> &kernel() { return kernel_; }

    // Const reference for member variables.
    const TVec<Scalar> &residual() const { return residual_; }
    const TMat<Scalar> &information() const { return information_; }
    const TMat<Scalar> &observation() const { return observation_; }
    const std::unique_ptr<Kernel<Scalar>> &kernel() const { return kernel_; }

    // Use string to represent edge type.
    virtual std::string GetType() { return std::string("Basic Edge"); }

    // Check if this edge valid.
    bool SelfCheck();

    // Compute Mahalanobis distance of residual.
    Scalar CalculateSquaredResidual() const { return residual_.transpose() * information_ * residual_; }

    // Compute residual and jacobians for each vertex. These operations should be defined by subclass.
    virtual void ComputeResidual() { residual_.setZero(); }
    virtual void ComputeJacobians() {
        for (auto &jacobian : jacobians_) {
            jacobian.setZero();
        }
    }

private:
    // Global index for every vertex.
    static uint32_t global_id_;
    uint32_t id_ = 0;

    // Vertice combined with this edge, and jacobians for all vertices.
    std::vector<Vertex<Scalar> *> vertices_ = {};
    std::vector<TMat<Scalar>> jacobians_ = {};

    // Residual for this edge.
    TVec<Scalar> residual_ = TVec3<Scalar>::Zero();

    // Information matrix for residual.
    TMat<Scalar> information_ = TMat3<Scalar>::Identity();

    // Observation for this edge. This can be varible in ComputeResidual() by using SetObservation().
    TMat<Scalar> observation_ = TVec3<Scalar>::Zero();

    // Kernel function, nullptr default.
    std::unique_ptr<Kernel<Scalar>> kernel_ = std::make_unique<Kernel<Scalar>>();
};

/* Class Edge Definition. */
template <typename Scalar>
uint32_t Edge<Scalar>::global_id_ = 0;

}

#endif // end of _GENERAL_GRAPH_OPTIMIZOR_EDGE_H_
