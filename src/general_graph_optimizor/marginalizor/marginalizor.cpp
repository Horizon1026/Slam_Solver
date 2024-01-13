#include "marginalizor.h"
#include "slam_operations.h"
#include "log_report.h"

namespace SLAM_SOLVER {

/* Specialized Template Class Declaration. */
template class Marginalizor<float>;
template class Marginalizor<double>;

/* Class Marginalizor Definition. */
template <typename Scalar>
bool Marginalizor<Scalar>::Marginalize(std::vector<Vertex<Scalar> *> &vertices,
                                       bool use_prior) {
    RETURN_FALSE_IF(problem_ == nullptr);

    // Sort all vertices, determine their location in incremental function.
    SortVerticesToBeMarged(vertices);
    problem_->SortVertices(false);

    // Linearize the non-linear problem, construct incremental function.
    ConstructInformation(use_prior);

    // Marginalize sparse vertices.
    MarginalizeSparseVertices();

    // Create information by schur complement.
    const int32_t reverse = this->problem()->full_size_of_dense_vertices() - size_of_vertices_need_marge_;
    const int32_t marg = size_of_vertices_need_marge_;
    CreatePriorInformation(reverse, marg);

    return true;
}

template <typename Scalar>
bool Marginalizor<Scalar>::Marginalize(TMat<Scalar> &hessian,
                                       TVec<Scalar> &bias,
                                       uint32_t row_index,
                                       uint32_t dimension) {
    RETURN_FALSE_IF(hessian.rows() != hessian.cols() || hessian.rows() != bias.rows());

    RETURN_FALSE_IF(!MoveMatrixBlocksNeedMarginalization(hessian, bias, row_index, dimension));

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
        CONTINUE_IF(vertex_to_be_marged == dense_vertices.rend());

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
void Marginalizor<Scalar>::CreatePriorInformation(int32_t reverse_size, int32_t marge_size) {
    switch (options_.kSortDirection) {
        // [ Hrr Hrm ] [ br ]
        // [ Hmr Hmm ] [ bm ]
        case SortMargedVerticesDirection::kSortAtBack: {
            TMat<Scalar> &&Hrr = reverse_hessian_.block(0, 0, reverse_size, reverse_size);
            TMat<Scalar> &&Hrm = reverse_hessian_.block(0, reverse_size, reverse_size, marge_size);
            TMat<Scalar> &&Hmr = reverse_hessian_.block(reverse_size, 0, marge_size, reverse_size);
            TMat<Scalar> Hmm = 0.5 * (reverse_hessian_.block(reverse_size, reverse_size, marge_size, marge_size) +
                reverse_hessian_.block(reverse_size, reverse_size, marge_size, marge_size).transpose());
            TVec<Scalar> &&br = reverse_bias_.head(reverse_size);
            TVec<Scalar> &&bm = reverse_bias_.tail(marge_size);

            ComputePriorBySchurComplement(Hrr, Hrm, Hmr, Hmm, br, bm);
            break;
        }

        // [ Hmm Hmr ] [ bm ]
        // [ Hrm Hrr ] [ br ]
        case SortMargedVerticesDirection::kSortAtFront:
        default: {
            TMat<Scalar> &&Hrr = reverse_hessian_.block(marge_size, marge_size, reverse_size, reverse_size);
            TMat<Scalar> &&Hrm = reverse_hessian_.block(marge_size, 0, reverse_size, marge_size);
            TMat<Scalar> &&Hmr = reverse_hessian_.block(0, marge_size, marge_size, reverse_size);
            TMat<Scalar> Hmm = 0.5 * (reverse_hessian_.block(0, 0, marge_size, marge_size) +
                reverse_hessian_.block(0, 0, marge_size, marge_size).transpose());
            TVec<Scalar> &&br = reverse_bias_.tail(reverse_size);
            TVec<Scalar> &&bm = reverse_bias_.head(marge_size);

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
    RETURN_IF(this->problem() == nullptr);
    auto &prior_hessian = this->problem()->prior_hessian();
    auto &prior_bias = this->problem()->prior_bias();
    auto &prior_jacobian = this->problem()->prior_jacobian();
    auto &prior_jacobian_t_inv = this->problem()->prior_jacobian_t_inv();
    auto &prior_residual = this->problem()->prior_residual();

    // Compute schur complement.
    SchurComplement(Hrr, Hrm, Hmr, Hmm, br, bm, prior_hessian, prior_bias);

    // Decompose prior hessian matrix and bias vector.
    DecomposeHessianAndBias(prior_hessian, prior_bias, prior_jacobian, prior_residual, prior_jacobian_t_inv);
}

template <typename Scalar>
void Marginalizor<Scalar>::SchurComplement(const TMat<Scalar> &Hrr,
                                           const TMat<Scalar> &Hrm,
                                           const TMat<Scalar> &Hmr,
                                           const TMat<Scalar> &Hmm,
                                           const TVec<Scalar> &br,
                                           const TVec<Scalar> &bm,
                                           TMat<Scalar> &hessian,
                                           TVec<Scalar> &bias) const {
    TMat<Scalar> Hmm_inv = SLAM_UTILITY::Utility::Inverse(Hmm);
    TMat<Scalar> Hrm_Hmm_inv = Hrm * Hmm_inv;
    hessian = Hrr - Hrm_Hmm_inv * Hmr;
    bias = br - Hrm_Hmm_inv * bm;
}

// Decompose hessian and bias to be jacobian and residual.
template <typename Scalar>
void Marginalizor<Scalar>::DecomposeHessianAndBias(TMat<Scalar> &hessian,
                                                   TVec<Scalar> &bias,
                                                   TMat<Scalar> &jacobian,
                                                   TVec<Scalar> &residual,
                                                   TMat<Scalar> &jacobian_t_inv) {
    // Decompose prior hessian matrix.
    Eigen::SelfAdjointEigenSolver<TMat<Scalar>> saes(hessian);
    const TVec<Scalar> S = TVec<Scalar>((saes.eigenvalues().array() > kZero).select(saes.eigenvalues().array(), 0));
    const TVec<Scalar> S_inv = TVec<Scalar>((saes.eigenvalues().array() > kZero).select(saes.eigenvalues().array().inverse(), 0));
    const TVec<Scalar> S_sqrt = S.cwiseSqrt();
    const TVec<Scalar> S_inv_sqrt = S_inv.cwiseSqrt();

    // Calculate prior information, store them in graph problem.
    const TMat<Scalar> eigen_vectors = saes.eigenvectors().transpose();
    jacobian_t_inv = S_inv_sqrt.asDiagonal() * eigen_vectors;
    residual = - jacobian_t_inv * bias;
    jacobian = S_sqrt.asDiagonal() * eigen_vectors;
    hessian = jacobian.transpose() * jacobian;
    const TMat<Scalar> tmp_h = TMat<Scalar>((hessian.array().abs() > kZero).select(hessian.array(), 0));
    hessian = tmp_h;
}

// Discard specified cols and rows of hessian and bias.
template <typename Scalar>
bool Marginalizor<Scalar>::DiscardPriorInformation(TMat<Scalar> &hessian,
                                                   TVec<Scalar> &bias,
                                                   uint32_t row_index,
                                                   uint32_t dimension) {
    RETURN_FALSE_IF(hessian.rows() != hessian.cols() || hessian.rows() != bias.rows());
    RETURN_FALSE_IF(row_index + dimension > hessian.rows());

    // If elements to be discarded is right-bottom rows and cols, resize directly.
    const uint32_t size = hessian.rows() - dimension;
    if (row_index + dimension == hessian.rows()) {
        hessian.conservativeResize(size, size);
        bias.conservativeResize(size, 1);
        return true;
    }

    // Move rows to be discarded to the bottom of hessian matrix.
    const TMat<Scalar> temp_rows = hessian.block(row_index + dimension, 0, hessian.rows() - row_index - dimension, hessian.cols());
    hessian.block(row_index, 0, temp_rows.rows(), temp_rows.cols()) = temp_rows;
    const TMat<Scalar> temp_cols = hessian.block(0, row_index + dimension, hessian.rows(), hessian.cols() - row_index - dimension);
    hessian.block(0, row_index, temp_cols.rows(), temp_cols.cols()) = temp_cols;
    hessian.conservativeResize(size, size);

    // Ditto bias vector.
    const TVec<Scalar> temp_tail = bias.segment(row_index + dimension, bias.rows() - row_index - dimension);
    bias.segment(row_index, temp_tail.rows()) = temp_tail;
    bias.conservativeResize(size, 1);

    return true;
}

    // Move the matrix block which needs to be margnalized to the bound of matrix.
template <typename Scalar>
bool Marginalizor<Scalar>::MoveMatrixBlocksNeedMarginalization(TMat<Scalar> &hessian,
                                                               TVec<Scalar> &bias,
                                                               uint32_t row_index,
                                                               uint32_t dimension) {
    RETURN_FALSE_IF(hessian.rows() != hessian.cols() || hessian.rows() != bias.rows());
    RETURN_FALSE_IF(row_index + dimension > hessian.rows());

    return true;
}

}
