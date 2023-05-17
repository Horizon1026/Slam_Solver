#ifndef _GENERAL_GRAPH_OPTIMIZOR_MARGINALIZOR_H_
#define _GENERAL_GRAPH_OPTIMIZOR_MARGINALIZOR_H_

#include "datatype_basic.h"
#include "math_kinematics.h"
#include "log_report.h"

#include "vertex.h"
#include "graph.h"

#include "vector"

namespace SLAM_SOLVER {

enum SortMargedVerticesDirection : uint8_t {
    kSortAtFront = 0,
    kSortAtBack = 1,
};

template <typename Scalar>
struct MargOptions {
    SortMargedVerticesDirection kSortDirection = SortMargedVerticesDirection::kSortAtFront;
};

/* Class Marginalizor Declaration. */
template <typename Scalar>
class Marginalizor {

public:
    Marginalizor() = default;
    virtual ~Marginalizor() = default;

    // Marginalize graph optimization problem.
    bool Marginalize(std::vector<Vertex<Scalar> *> &vertices,
                     bool use_prior = true);

    // Sort vertices to be marged to the front or back of vertices vector.
    // Keep the other vertices the same order.
    virtual void SortVerticesToBeMarged(std::vector<Vertex<Scalar> *> &vertices);

    // Construct information.
    virtual void ConstructInformation(bool use_prior = true);

    // Marginalize sparse vertices in information.
    virtual void MarginalizeSparseVertices();

    // Create prior information, and store them in graph problem.
    virtual void CreatePriorInformation();

    // Reference for member varibles.
    MargOptions<Scalar> &options() { return options_; }
    Graph<Scalar> *&problem() { return problem_; }

private:
    // Compute prior information with schur complement.
    void ComputePriorBySchurComplement(const TMat<Scalar> &Hrr,
                                       const TMat<Scalar> &Hrm,
                                       const TMat<Scalar> &Hmr,
                                       const TMat<Scalar> &Hmm,
                                       const TVec<Scalar> &br,
                                       const TVec<Scalar> &bm);

private:
    // General options for marginalizor.
    MargOptions<Scalar> options_;

    // The graph optimization problem to be marged.
    Graph<Scalar> *problem_ = nullptr;

    // The size of vertices needing to be marged.
    int32_t size_of_vertices_need_marge_ = 0;

    // Parameters of schur complement.
    TMat<Scalar> reverse_hessian_;
    TVec<Scalar> reverse_bias_;

};

/* Class Marginalizor Definition. */
template <typename Scalar>
bool Marginalizor<Scalar>::Marginalize(std::vector<Vertex<Scalar> *> &vertices,
                                       bool use_prior) {
    if (problem_ == nullptr) {
        return false;
    }

    // Sort all vertices, determine their location in incremental function.
    SortVerticesToBeMarged(vertices);
    problem_->SortVertices(false);

    // Linearize the non-linear problem, construct incremental function.
    ConstructInformation(use_prior);

    // Marginalize sparse vertices.
    MarginalizeSparseVertices();

    // Create information by schur complement.
    CreatePriorInformation();

    return true;
}

// Sort vertices to be marged to the front or back of vertices vector.
// Keep the other vertices the same order.
template <typename Scalar>
void Marginalizor<Scalar>::SortVerticesToBeMarged(std::vector<Vertex<Scalar> *> &vertices) {
    auto &dense_vertices = problem_->dense_vertices();
    size_of_vertices_need_marge_ = 0;

    for (const auto &vertex : vertices) {
        // Statis full size of vertices need to be marged.
        size_of_vertices_need_marge_ += vertex->GetIncrementDimension();

        // Find the vertex to be marged from back to front.
        auto vertex_to_be_marged = std::find(dense_vertices.rbegin(), dense_vertices.rend(), vertex);
        if (vertex_to_be_marged == dense_vertices.rend()) {
            continue;
        }

        // If vertex is found, compute its index in std::vector<>.
        const int32_t idx = std::distance(vertex_to_be_marged, dense_vertices.rend()) - 1;

        switch (options_.kSortDirection) {
            case SortMargedVerticesDirection::kSortAtBack: {
                // Move the vertex to be marged to the back of this std::vector<>.
                const int32_t max_idx = dense_vertices.size() - 1;
                for (int32_t i = idx; i < max_idx; ++i) {
                    const auto temp_vertex = dense_vertices[i];
                    dense_vertices[i] = dense_vertices[i + 1];
                    dense_vertices[i + 1] = temp_vertex;
                }

                break;
            }
            case SortMargedVerticesDirection::kSortAtFront:
            default: {
                // Move the vertex to be marged to the front of this std::vector<>.
                for (int32_t i = idx; i > 0; --i) {
                    const auto temp_vertex = dense_vertices[i];
                    dense_vertices[i] = dense_vertices[i - 1];
                    dense_vertices[i - 1] = temp_vertex;
                }
                break;
            }
        }
    }
}

// Construct information.
template <typename Scalar>
void Marginalizor<Scalar>::ConstructInformation(bool use_prior) {
    problem_->ComputeResidualForAllEdges(use_prior);
    problem_->ComputeJacobiansForAllEdges();
    problem_->ConstructFullSizeHessianAndBias(use_prior);
}

// Marginalize sparse vertices in information.
template <typename Scalar>
void Marginalizor<Scalar>::MarginalizeSparseVertices() {
    const int32_t marg = this->problem()->full_size_of_sparse_vertices();
    if (marg) {
        this->problem()->MarginalizeSparseVerticesInHessianAndBias(reverse_hessian_, reverse_bias_);
    } else {
        reverse_hessian_ = this->problem()->hessian();
        reverse_bias_ = this->problem()->bias();
    }
}

// Create prior information, and store them in graph problem.
template <typename Scalar>
void Marginalizor<Scalar>::CreatePriorInformation() {
    const int32_t dense_size = this->problem()->full_size_of_dense_vertices();
    const int32_t reverse = dense_size - size_of_vertices_need_marge_;
    const int32_t marg = size_of_vertices_need_marge_;

    switch (options_.kSortDirection) {
        // [ Hrr Hrm ] [ br ]
        // [ Hmr Hmm ] [ bm ]
        case SortMargedVerticesDirection::kSortAtBack: {
            TMat<Scalar> &&Hrr = this->problem()->hessian().block(0, 0, reverse, reverse);
            TMat<Scalar> &&Hrm = this->problem()->hessian().block(0, reverse, reverse, marg);
            TMat<Scalar> &&Hmr = this->problem()->hessian().block(reverse, 0, marg, reverse);
            TMat<Scalar> &&Hmm = this->problem()->hessian().block(reverse, reverse, marg, marg);
            TVec<Scalar> &&br = this->problem()->bias().head(reverse);
            TVec<Scalar> &&bm = this->problem()->bias().tail(marg);

            ComputePriorBySchurComplement(Hrr, Hrm, Hmr, Hmm, br, bm);
            break;
        }

        // [ Hmm Hmr ] [ bm ]
        // [ Hrm Hrr ] [ br ]
        case SortMargedVerticesDirection::kSortAtFront:
        default: {
            TMat<Scalar> &&Hrr = this->problem()->hessian().block(marg, marg, reverse, reverse);
            TMat<Scalar> &&Hrm = this->problem()->hessian().block(marg, 0, reverse, marg);
            TMat<Scalar> &&Hmr = this->problem()->hessian().block(0, marg, marg, reverse);
            TMat<Scalar> &&Hmm = this->problem()->hessian().block(0, 0, marg, marg);
            TVec<Scalar> &&br = this->problem()->bias().tail(reverse);
            TVec<Scalar> &&bm = this->problem()->bias().head(marg);

            ComputePriorBySchurComplement(Hrr, Hrm, Hmr, Hmm, br, bm);
            break;
        }
    }
}

// Compute prior information with schur complement.
template <typename Scalar>
void Marginalizor<Scalar>::ComputePriorBySchurComplement(const TMat<Scalar> &Hrr,
                                                         const TMat<Scalar> &Hrm,
                                                         const TMat<Scalar> &Hmr,
                                                         const TMat<Scalar> &Hmm,
                                                         const TVec<Scalar> &br,
                                                         const TVec<Scalar> &bm) {
    auto &prior_hessian = this->problem()->prior_hessian();
    auto &prior_bias = this->problem()->prior_bias();
    auto &prior_jacobian = this->problem()->prior_jacobian();
    auto &prior_jacobian_t_inv = this->problem()->prior_jacobian_t_inv();
    auto &prior_residual = this->problem()->prior_residual();

    // Compute schur complement.
    TMat<Scalar> Hmm_inv = SLAM_UTILITY::Utility::Inverse(Hmm);
    TMat<Scalar> Hrm_Hmm_inv = Hrm * Hmm_inv;
    prior_hessian = Hrr - Hrm_Hmm_inv * Hmr;
    prior_bias = br - Hrm_Hmm_inv * bm;

    // Decompose prior hessian matrix.
    Eigen::SelfAdjointEigenSolver<TMat<Scalar>> saes(prior_hessian);
    TVec<Scalar> S = TVec<Scalar>((saes.eigenvalues().array() > kZero).select(saes.eigenvalues().array(), 0));
    TVec<Scalar> S_inv = TVec<Scalar>((saes.eigenvalues().array() > kZero).select(saes.eigenvalues().array().inverse(), 0));
    TVec<Scalar> S_sqrt = S.cwiseSqrt();
    TVec<Scalar> S_inv_sqrt = S_inv.cwiseSqrt();

    // Calculate prior information, store them in graph problem.
    TMat<Scalar> eigen_vectors_t = saes.eigenvectors().transpose();
    prior_jacobian_t_inv = S_inv_sqrt.asDiagonal() * eigen_vectors_t;
    prior_residual = -prior_jacobian_t_inv * prior_bias;
    prior_jacobian = S_sqrt.asDiagonal() * eigen_vectors_t;
    prior_hessian = prior_jacobian.transpose() * prior_jacobian;
    TMat<Scalar> tmp_h = TMat<Scalar>((prior_hessian.array().abs() > kZero).select(prior_hessian.array(), 0));
    prior_hessian = tmp_h;
}

}

#endif // end of _GENERAL_GRAPH_OPTIMIZOR_MARGINALIZOR_H_
