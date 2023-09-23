#include "graph.h"

#ifdef ENABLE_TBB_PARALLEL
#pragma message("tbb/tbb.h is included.")
#include "tbb/tbb.h"
#endif

namespace SLAM_SOLVER {

/* Specialized Template Class Declaration. */
template class Graph<float>;
template class Graph<double>;

/* Class Graph Definition. */
template <typename Scalar>
Graph<Scalar>::Graph(uint32_t reserved_vertex_num) {
    dense_vertices_.reserve(reserved_vertex_num);
    sparse_vertices_.reserve(reserved_vertex_num);
    edges_.reserve(reserved_vertex_num);

    Clear();
}

// Clear everything of this graph.
template <typename Scalar>
void Graph<Scalar>::Clear() {
    dense_vertices_.clear();
    sparse_vertices_.clear();
    edges_.clear();

    full_size_of_dense_vertices_ = 0;
    full_size_of_sparse_vertices_ = 0;
    full_size_of_residuals_ = 0;
}

// Add vertices and edges for this graph.
template <typename Scalar>
bool Graph<Scalar>::AddVertex(Vertex<Scalar> *vertex, bool is_dense_vertex) {
    if (vertex == nullptr) {
        return false;
    }

    if (is_dense_vertex) {
        dense_vertices_.emplace_back(vertex);
    } else {
        sparse_vertices_.emplace_back(vertex);
    }
    return true;
}

template <typename Scalar>
bool Graph<Scalar>::AddVertexWithCheck(Vertex<Scalar> *vertex, bool is_dense_vertex) {
    if (std::find(dense_vertices_.cbegin(), dense_vertices_.cend(), vertex) != dense_vertices_.cend()) {
        return false;
    }
    if (std::find(sparse_vertices_.cbegin(), sparse_vertices_.cend(), vertex) != sparse_vertices_.cend()) {
        return false;
    }

    return AddVertex(vertex, is_dense_vertex);
}

template <typename Scalar>
bool Graph<Scalar>::AddEdge(Edge<Scalar> *edge) {
    if (edge == nullptr) {
        return false;
    }

    edges_.emplace_back(edge);
    return true;
}

template <typename Scalar>
bool Graph<Scalar>::AddEdgeWithCheck(Edge<Scalar> *edge) {
    if (std::find(edges_.cbegin(), edges_.cend(), edge) == edges_.cend()) {
        return false;
    }

    return AddEdge(edge);
}

// Sort all vertices in incremental function.
template <typename Scalar>
void Graph<Scalar>::SortVertices(bool statis_size_of_residual) {
    full_size_of_dense_vertices_ = 0;
    for (auto &item : dense_vertices_) {
        item->ColIndex() = full_size_of_dense_vertices_;
        full_size_of_dense_vertices_ += item->GetIncrementDimension();
    }

    full_size_of_sparse_vertices_ = 0;
    for (auto &item : sparse_vertices_) {
        item->ColIndex() = full_size_of_sparse_vertices_ + full_size_of_dense_vertices_;
        full_size_of_sparse_vertices_ += item->GetIncrementDimension();
    }

    // Size of residual is only used for Jacobian. If only use hessian and bias, it is not necessay.
    if (statis_size_of_residual) {
        full_size_of_residuals_ = 0;
        for (auto &item : edges_) {
            full_size_of_residuals_ += item->residual().size();
        }
    }
}

// Update all vertices.
template <typename Scalar>
void Graph<Scalar>::UpdateAllVertices(const TVec<Scalar> &delta_x) {
#ifdef ENABLE_TBB_PARALLEL
    tbb::parallel_for(tbb::blocked_range<uint32_t>(0, dense_vertices_.size()),
        [&] (tbb::blocked_range<uint32_t> range) {
            for (uint32_t i = range.begin(); i < range.end(); ++i) {
                auto &vertex = dense_vertices_[i];
                if (vertex->IsFixed()) {
                    continue;
                }
                vertex->BackupParam();
                const int32_t index = vertex->ColIndex();
                const int32_t dim = vertex->GetIncrementDimension();
                vertex->UpdateParam(delta_x.segment(index, dim));
            }
        }
    );

    tbb::parallel_for(tbb::blocked_range<uint32_t>(0, sparse_vertices_.size()),
        [&] (tbb::blocked_range<uint32_t> range) {
            for (uint32_t i = range.begin(); i < range.end(); ++i) {
                auto &vertex = sparse_vertices_[i];
                if (vertex->IsFixed()) {
                    continue;
                }
                vertex->BackupParam();
                const int32_t index = vertex->ColIndex();
                const int32_t dim = vertex->GetIncrementDimension();
                vertex->UpdateParam(delta_x.segment(index, dim));
            }
        }
    );
#else // ENABLE_TBB_PARALLEL
    for (auto &vertex : dense_vertices_) {
        if (vertex->IsFixed()) {
            continue;
        }
        vertex->BackupParam();
        const int32_t index = vertex->ColIndex();
        const int32_t dim = vertex->GetIncrementDimension();
        vertex->UpdateParam(delta_x.segment(index, dim));
    }

    for (auto &vertex : sparse_vertices_) {
        if (vertex->IsFixed()) {
            continue;
        }
        vertex->BackupParam();
        const int32_t index = vertex->ColIndex();
        const int32_t dim = vertex->GetIncrementDimension();
        vertex->UpdateParam(delta_x.segment(index, dim));
    }
#endif // end of ENABLE_TBB_PARALLEL
}

// Roll back all vertices.
template <typename Scalar>
void Graph<Scalar>::RollBackAllVertices() {
    for (auto &vertex : dense_vertices_) {
        if (vertex->IsFixed()) {
            continue;
        }

        vertex->RollbackParam();
    }

    for (auto &vertex : sparse_vertices_) {
        if (vertex->IsFixed()) {
            continue;
        }

        vertex->RollbackParam();
    }
}

// Compute residual for all edges.
template <typename Scalar>
Scalar Graph<Scalar>::ComputeResidualForAllEdges(bool use_prior) {
#ifdef ENABLE_TBB_PARALLEL
    Scalar sum_cost = tbb::parallel_reduce(tbb::blocked_range<uint32_t>(0, edges_.size()), Scalar(0),
        [&] (tbb::blocked_range<uint32_t> range, Scalar sub_sum_cost) {
            for (uint32_t i = range.begin(); i < range.end(); ++i) {
                auto &edge = edges_[i];
                edge->ComputeResidual();
                const Scalar x = edge->CalculateSquaredResidual();
                edge->kernel()->Compute(x);
                sub_sum_cost += edge->kernel()->y(0);
            }
            return sub_sum_cost;
        }, std::plus<Scalar>()
    );
#else // ENABLE_TBB_PARALLEL
    Scalar sum_cost = 0;
    for (auto &edge : edges_) {
        edge->ComputeResidual();
        const Scalar x = edge->CalculateSquaredResidual();
        edge->kernel()->Compute(x);
        sum_cost += edge->kernel()->y(0);
    }
#endif // end of ENABLE_TBB_PARALLEL

    // Prior information is decomposed as sqrt(S) * r, so squredNorm means r.t * S * r.
    if (use_prior && prior_residual_.rows() > 0) {
        sum_cost += prior_residual_.squaredNorm();
    }

    return sum_cost;
}

// Compute jacobian for all vertices and edges
template <typename Scalar>
void Graph<Scalar>::ComputeJacobiansForAllEdges() {
#ifdef ENABLE_TBB_PARALLEL
    tbb::parallel_for(tbb::blocked_range<uint32_t>(0, edges_.size()),
        [&] (tbb::blocked_range<uint32_t> range) {
            for (uint32_t i = range.begin(); i < range.end(); ++i) {
                edges_[i]->ComputeJacobians();
            }
        }
    );
#else // ENABLE_TBB_PARALLEL
    for (auto &edge : edges_) {
        edge->ComputeJacobians();
    }
#endif // end of ENABLE_TBB_PARALLEL
}

// Construct incremental function with full size.
template <typename Scalar>
void Graph<Scalar>::ConstructFullSizeJacobianAndResidual(bool use_prior) {

}

// Construct incremental function with full size.
template <typename Scalar>
void Graph<Scalar>::ConstructFullSizeHessianAndBias(bool use_prior) {
    const int32_t size = full_size_of_dense_vertices_ + full_size_of_sparse_vertices_;
    hessian_.setZero(size, size);
    bias_.setZero(size);

    // Preallocate memory for temp variables.
    TMat<Scalar> Jt_S_w;
    TMat<Scalar> sub_hessian;

    // Traverse all edges, use residual and jacobians to construct full size hessian and bias.
    for (const auto &edge : edges_) {
        const uint32_t vertex_num = edge->GetVertexNum();
        const auto &vertices_in_edge = edge->GetVertices();
        const auto &jacobians_in_edge = edge->GetJacobians();

        for (uint32_t i = 0; i < vertex_num; ++i) {
            if (vertices_in_edge[i]->IsFixed()) {
                continue;
            }
            const int32_t index_i = vertices_in_edge[i]->ColIndex();
            const int32_t dim_i = vertices_in_edge[i]->GetIncrementDimension();

            // Precompute J.t * S, which is useful to calculate J.t * S * S and J.t * S * r
            // Weight of sub_hessian and sub_bias is w = rho'(r.t * S * r)
            Jt_S_w = jacobians_in_edge[i].transpose() * edge->information() * edge->kernel()->y(1);

            // Fill bias with J.t * S * w * r.
            bias_.segment(index_i, dim_i).noalias() -= Jt_S_w * edge->residual();

            // Fill hessian diagnal with J.t * S * w * J.
            sub_hessian = Jt_S_w * jacobians_in_edge[i];
            hessian_.block(index_i, index_i, dim_i, dim_i).noalias() += sub_hessian;

            for (uint32_t j = i + 1; j < vertex_num; ++j) {
                if (vertices_in_edge[j]->IsFixed()) {
                    continue;
                }
                const int32_t index_j = vertices_in_edge[j]->ColIndex();
                const int32_t dim_j = vertices_in_edge[j]->GetIncrementDimension();

                // Fill hessian diagnal with J.t * S * w * J.
                sub_hessian = Jt_S_w * jacobians_in_edge[j];
                hessian_.block(index_i, index_j, dim_i, dim_j).noalias() += sub_hessian;
                hessian_.block(index_j, index_i, dim_j, dim_i).noalias() += sub_hessian.transpose();
            }
        }
    }

    // Add prior information on incremental function, if configed to use prior.
    if (use_prior && prior_hessian_.rows() > 0) {
        // Adjust prior information.
        for (const auto &vertex : dense_vertices_) {
            if (vertex->IsFixed()) {
                const int32_t index = vertex->ColIndex();
                const int32_t dim = vertex->GetIncrementDimension();
                if (index + size > prior_hessian_.cols()) {
                    continue;
                }

                // If vertex is fixed, its prior information should be cleaned.
                prior_hessian_.block(index, 0, dim, prior_hessian_.cols()).setZero();
                prior_hessian_.block(0, index, prior_hessian_.rows(), dim).setZero();
                prior_bias_.segment(index, dim).setZero();
            }
        }

        // Add prior information on incremantal function.
        hessian_.topLeftCorner(prior_hessian_.rows(), prior_hessian_.cols()).noalias() += prior_hessian_;
        bias_.head(prior_bias_.rows()).noalias() += prior_bias_;
    }
}

// Marginalize sparse vertices in incremental function.
template <typename Scalar>
void Graph<Scalar>::MarginalizeSparseVerticesInHessianAndBias(TMat<Scalar> &hessian, TVec<Scalar> &bias) {
    // Dense vertices should be kept, and sparse vertices should be marginalized.
    const uint32_t reverse = full_size_of_dense_vertices_;
    const uint32_t marg = full_size_of_sparse_vertices_;

    // Calculate inverse of Hmm to get Hrm * Hmm_inv.
    TMat<Scalar> Hrm_Hmm_inv = TMat<Scalar>::Zero(reverse, marg);
#ifdef ENABLE_TBB_PARALLEL
    tbb::parallel_for(tbb::blocked_range<uint32_t>(0, sparse_vertices_.size()),
        [&] (tbb::blocked_range<uint32_t> range) {
            for (uint32_t i = range.begin(); i < range.end(); ++i) {
                auto &vertex = sparse_vertices_[i];
                const int32_t index = vertex->ColIndex();
                const int32_t dim = vertex->GetIncrementDimension();

                Hrm_Hmm_inv.block(0, index - reverse, reverse, dim) =
                    hessian_.block(0, index, reverse, dim) *
                    hessian_.block(index, index, dim, dim).inverse();
            }
        }
    );
#else // ENABLE_TBB_PARALLEL
    for (const auto &vertex : sparse_vertices_) {
        const int32_t index = vertex->ColIndex();
        const int32_t dim = vertex->GetIncrementDimension();

        Hrm_Hmm_inv.block(0, index - reverse, reverse, dim) =
            hessian_.block(0, index, reverse, dim) *
            hessian_.block(index, index, dim, dim).inverse();
    }
#endif // end of ENABLE_TBB_PARALLEL

    // Calculate schur complement.
    // subH = Hrr - Hrm_Hmm_inv * Hmr.
    // subb = br - Hrm_Hmm_inv * bm.
    hessian = hessian_.block(0, 0, reverse, reverse) - Hrm_Hmm_inv * hessian_.block(reverse, 0, marg, reverse);
    bias = bias_.segment(0, reverse) - Hrm_Hmm_inv * bias_.segment(reverse, marg);
}

template <typename Scalar>
void Graph<Scalar>::MarginalizeSparseVerticesInJacobianAndResidual(TMat<Scalar> &jacobian, TVec<Scalar> &residual) {

}

}
