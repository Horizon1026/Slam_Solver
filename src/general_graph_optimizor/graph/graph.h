#ifndef _GENERAL_GRAPH_OPTIMIZOR_GRAPH_H_
#define _GENERAL_GRAPH_OPTIMIZOR_GRAPH_H_

#include "basic_type.h"
#include "edge.h"
#include "vertex.h"

#include "memory"
#include "vector"

namespace SLAM_SOLVER {

/* Class Graph Declaration. */
template <typename Scalar>
class Graph {

public:
    Graph() { Graph(50); }
    explicit Graph(uint32_t reserved_vertex_num);
    virtual ~Graph() = default;

    void Clear();
    void VerticesInformation(bool show_sparse_vertices_info = false);
    void EdgesInformation();

    // Add vertices and edges for this graph.
    bool AddVertex(Vertex<Scalar> *vertex, bool is_dense_vertex = true);
    bool AddVertexWithCheck(Vertex<Scalar> *vertex, bool is_dense_vertex = true);
    bool AddEdge(Edge<Scalar> *edge);
    bool AddEdgeWithCheck(Edge<Scalar> *edge);

    // Get size of vertices and edges.
    int32_t VertexNum() const { return dense_vertices_.size() + sparse_vertices_.size(); }
    int32_t EdgeNum() const { return edges_.size(); }

    // Sort all vertices in incremental function.
    void SortVertices(bool statis_size_of_residual = false);

    // Update and roll back all vertices.
    void UpdateAllVertices(const TVec<Scalar> &delta_x);
    void RollBackAllVertices();

    // Compute residual for all edges and jacobian for all vertices and edges.
    Scalar ComputeResidualForAllEdges(bool use_prior = false);
    void ComputeJacobiansForAllEdges();

    // Construct incremental function with full size.
    void ConstructFullSizeJacobianAndResidual(bool use_prior = false);
    void ConstructFullSizeHessianAndBias(bool use_prior = false);

    // Marginalize sparse vertices in incremental function.
    void MarginalizeSparseVerticesInHessianAndBias(TMat<Scalar> &hessian, TVec<Scalar> &bias);
    void MarginalizeSparseVerticesInJacobianAndResidual(TMat<Scalar> &jacobian, TVec<Scalar> &residual);

    // Reference of member variables.
    std::vector<Vertex<Scalar> *> &dense_vertices() { return dense_vertices_; }
    std::vector<Vertex<Scalar> *> &sparse_vertices() { return sparse_vertices_; }
    std::vector<Edge<Scalar> *> &edges() { return edges_; }
    TMat<Scalar> &hessian() { return hessian_; }
    TVec<Scalar> &bias() { return bias_; }
    TMat<Scalar> &jacobian() { return jacobian_; }
    TVec<Scalar> &residual() { return residual_; }
    TMat<Scalar> &prior_hessian() { return prior_hessian_; }
    TVec<Scalar> &prior_bias() { return prior_bias_; }
    TMat<Scalar> &prior_jacobian() { return prior_jacobian_; }
    TMat<Scalar> &prior_jacobian_t_inv() { return prior_jacobian_t_inv_; }
    TVec<Scalar> &prior_residual() { return prior_residual_; }

    // Const reference of member variables.
    const std::vector<Vertex<Scalar> *> &dense_vertices() const { return dense_vertices_; }
    const std::vector<Vertex<Scalar> *> &sparse_vertices() const { return sparse_vertices_; }
    const std::vector<Edge<Scalar> *> &edges() const { return edges_; }
    const int32_t &full_size_of_dense_vertices() const { return full_size_of_dense_vertices_; }
    const int32_t &full_size_of_sparse_vertices() const { return full_size_of_sparse_vertices_; }
    const int32_t &full_size_of_residuals() const { return full_size_of_residuals_; }
    const TMat<Scalar> &hessian() const { return hessian_; }
    const TVec<Scalar> &bias() const { return bias_; }
    const TMat<Scalar> &jacobian() const { return jacobian_; }
    const TVec<Scalar> &residual() const { return residual_; }
    const TMat<Scalar> &prior_hessian() const { return prior_hessian_; }
    const TVec<Scalar> &prior_bias() const { return prior_bias_; }
    const TMat<Scalar> &prior_jacobian() const { return prior_jacobian_; }
    const TMat<Scalar> &prior_jacobian_t_inv() const { return prior_jacobian_t_inv_; }
    const TVec<Scalar> &prior_residual() const { return prior_residual_; }

private:
    // Manage vertices and edges in this graph.
    // When construct incremental function, sparse vertices will be marginalized.
    std::vector<Vertex<Scalar> *> dense_vertices_ = {};
    std::vector<Vertex<Scalar> *> sparse_vertices_ = {};
    std::vector<Edge<Scalar> *> edges_ = {};

    // Statis size of vertices with two different types.
    int32_t full_size_of_dense_vertices_ = 0;
    int32_t full_size_of_sparse_vertices_ = 0;
    int32_t full_size_of_residuals_ = 0;

    // Hessian matrix and bias vector in incremantal function.
    TMat<Scalar> hessian_ = TMat3<Scalar>::Zero();
    TVec<Scalar> bias_ = TVec3<Scalar>::Zero();

    // Jacobian matrix and residual vector in incremental function.
    TMat<Scalar> jacobian_ = TMat3<Scalar>::Identity();
    TVec<Scalar> residual_ = TVec3<Scalar>::Zero();

    // Prior information and useful backup.
    TMat<Scalar> prior_hessian_;
    TVec<Scalar> prior_bias_;
    TMat<Scalar> prior_jacobian_;
    TMat<Scalar> prior_jacobian_t_inv_;
    TVec<Scalar> prior_residual_;
    TVec<Scalar> backup_prior_bias_;
    TVec<Scalar> backup_prior_residual_;
};

}  // namespace SLAM_SOLVER

#endif
