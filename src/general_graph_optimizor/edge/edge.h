#ifndef _GENERAL_GRAPH_OPTIMIZOR_EDGE_H_
#define _GENERAL_GRAPH_OPTIMIZOR_EDGE_H_

#include "datatype_basic.h"
#include "log_api.h"
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
    const Vertex<Scalar> *GetVertex(uint32_t index) const { return vertices_[index]; }
    const std::vector<Vertex<Scalar> *> &GetVertices() const { return vertices_; }
    uint32_t GetVertexNum() const { return vertices_.size(); }

    // Set and get jacobians of each vertex of this edge.
    void SetJacobian(const TMat<Scalar> &jacobian, uint32_t index) { jacobians_[index] = jacobian; }
    const TMat<Scalar> &GetJacobian(uint32_t index) const { return jacobians_[index]; }

    // Reference of residual, information and observation for this edge.
    TVec<Scalar> &residual() { return residual_; }
    TMat<Scalar> &information() { return information_; }
    TMat<Scalar> &observation() { return observation_; }

    // Kernel function ptr reference.
    std::unique_ptr<Kernel<Scalar>> &kernel() { return kernel_; }

    // Use string to represent edge type.
    virtual std::string GetType() { return std::string("Basic Edge"); }

    // Check if this edge valid.
    bool SelfCheck();

    // Compute Mahalanobis distance of residual.
    Scalar CalculateSquaredResidual() const { return residual_.transpose() * information * residual_; }

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
    TMat<Scalar> information_ = TMat3<Scalar>::Zero();

    // Observation for this edge. This can be varible in ComputeResidual() by using SetObservation().
    TMat<Scalar> observation_ = TVec3<Scalar>::Zero();

    // Kernel function, nullptr default.
    std::unique_ptr<Kernel<Scalar>> kernel_ = std::make_unique<Kernel<Scalar>>();
};

/* Class Edge Definition. */
template <typename Scalar>
uint32_t Edge<Scalar>::global_id_ = 0;

template <typename Scalar>
Edge<Scalar>::Edge(int32_t residual_dim, int32_t vertex_num) {
    // Resize size of residual and jacobian.
    if (residual_.size() != residual_dim) {
        residual_.resize(residual_dim);
        information_.setIdentity(residual_dim, residual_dim);
    }
    vertices_.resize(vertex_num, nullptr);
    jacobians_.resize(vertex_num);

    // Set index.
    ++Edge<Scalar>::global_id_;
    id_ = Edge<Scalar>::global_id_;
}

// Set vertex of this edge.
template <typename Scalar>
bool Edge<Scalar>::SetVertex(Vertex<Scalar> * vertex, uint32_t index) {
    if (vertex == nullptr || index > vertices_.size() - 1) {
        return false;
    }

    vertices_[index] = vertex;
    jacobians_[index].setZero(residual_.rows(), vertices_[index]->GetIncrementDimension());
    return true;
}

template <typename Scalar>
bool Edge<Scalar>::SetVertices(const std::vector<Vertex<Scalar> *> &vertices) {
    if (vertices_.size() != vertices.size()) {
        return false;
    }

    for (const auto &vertex : vertices) {
        if (vertex == nullptr) {
            return false;
        }
    }

    // Copy vertices and resize jacobians.
    for (uint32_t i = 0; i < vertices_.size(); ++i) {
        vertices_[i] = vertices[i];
        jacobians_[i].setZero(residual_.rows(), vertices_[i]->GetIncrementDimension());
    }
    return true;
}

// Check if this edge valid.
template <typename Scalar>
bool Edge<Scalar>::SelfCheck() {
    if (vertices_.size() != jacobians_.size()) {
        LogError("[Edge] Vertices size is not equal to jacobians size.");
        return false;
    }

    if (residual_.rows() != information_.rows() || residual_.rows() != information_.cols()) {
        LogError("[Edge] Residual size doesn't match information matrix.");
        return false;
    }

    for (uint32_t i = 0; i < vertices_.size(); ++i) {
        if (vertices_[i] == nullptr) {
            LogError("[Edge] Vertex " << i << " is nullptr.");
            return false;
        }

        if (vertices_[i]->GetIncrementDimension() != jacobians_[i].cols()) {
            LogError("[Edge] Vertex " << i << " doesn't match its jacobian.");
            return false;
        }

        if (residual_.rows() != jacobians_[i].rows()) {
            LogError("[Edge] Residual size doesn't match jacobian.");
            return false;
        }
    }

    if (kernel_ == nullptr) {
        LogError("[Edge] Kernel function is nullptr.");
        return false;
    }

    return true;
}

}

#endif // end of _GENERAL_GRAPH_OPTIMIZOR_EDGE_H_
