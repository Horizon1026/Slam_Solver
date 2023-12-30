#include "marginalizor.h"

namespace SLAM_SOLVER {

/* Specialized Template Class Declaration. */
template class Marginalizor<float>;
template class Marginalizor<double>;

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
    cost_of_problem_ = problem_->ComputeResidualForAllEdges(use_prior);
    problem_->ComputeJacobiansForAllEdges();
    problem_->ConstructFullSizeHessianAndBias(use_prior);
}

// Marginalize sparse vertices in information.
template <typename Scalar>
void Marginalizor<Scalar>::MarginalizeSparseVertices() {
    const int32_t marg = this->problem()->full_size_of_sparse_vertices();
    if (marg > 0) {
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
            TMat<Scalar> &&Hrr = reverse_hessian_.block(0, 0, reverse, reverse);
            TMat<Scalar> &&Hrm = reverse_hessian_.block(0, reverse, reverse, marg);
            TMat<Scalar> &&Hmr = reverse_hessian_.block(reverse, 0, marg, reverse);
            TMat<Scalar> &&Hmm = reverse_hessian_.block(reverse, reverse, marg, marg);
            TVec<Scalar> &&br = reverse_bias_.head(reverse);
            TVec<Scalar> &&bm = reverse_bias_.tail(marg);

            ComputePriorBySchurComplement(Hrr, Hrm, Hmr, Hmm, br, bm);
            break;
        }

        // [ Hmm Hmr ] [ bm ]
        // [ Hrm Hrr ] [ br ]
        case SortMargedVerticesDirection::kSortAtFront:
        default: {
            TMat<Scalar> &&Hrr = reverse_hessian_.block(marg, marg, reverse, reverse);
            TMat<Scalar> &&Hrm = reverse_hessian_.block(marg, 0, reverse, marg);
            TMat<Scalar> &&Hmr = reverse_hessian_.block(0, marg, marg, reverse);
            TMat<Scalar> &&Hmm = reverse_hessian_.block(0, 0, marg, marg);
            TVec<Scalar> &&br = reverse_bias_.tail(reverse);
            TVec<Scalar> &&bm = reverse_bias_.head(marg);

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
