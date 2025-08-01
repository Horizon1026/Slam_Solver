#include "marginalizor.h"
#include "slam_operations.h"
#include "slam_log_reporter.h"

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
    problem_->SortVertices(false);

    // Linearize the non-linear problem, construct incremental function.
    ConstructInformation(use_prior);

    // Marginalize sparse vertices.
    MarginalizeSparseVertices();

    // Move the matrix block which needs to be marginalized to the bound of hessian.
    // Statis full size of vertices need to be marged.
    size_of_vertices_need_marge_ = 0;
    for (const auto &vertex: vertices) {
        RETURN_FALSE_IF(!MoveMatrixBlocksNeedMarginalization(reverse_hessian_, reverse_bias_, vertex->ColIndex(), vertex->GetIncrementDimension()));
        size_of_vertices_need_marge_ += vertex->GetIncrementDimension();
    }

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

    // Move the matrix block which needs to be marginalized to the bound of hessian.
    RETURN_FALSE_IF(!MoveMatrixBlocksNeedMarginalization(hessian, bias, row_index, dimension));

    // Only create prior hessian and bias. Store them in reverse_hessian/bias.
    const int32_t reverse = hessian.rows() - dimension;
    const int32_t marg = dimension;
    CreatePriorInformationOnlyHessianAndBias(hessian, bias, reverse, marg);

    // Generate output.
    hessian = reverse_hessian_;
    bias = reverse_bias_;

    return true;
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
            const TMat<Scalar> &&Hrr = reverse_hessian_.block(0, 0, reverse_size, reverse_size);
            const TMat<Scalar> &&Hrm = reverse_hessian_.block(0, reverse_size, reverse_size, marge_size);
            const TMat<Scalar> &&Hmr = reverse_hessian_.block(reverse_size, 0, marge_size, reverse_size);
            const TMat<Scalar> Hmm = 0.5 * (reverse_hessian_.block(reverse_size, reverse_size, marge_size, marge_size) +
                reverse_hessian_.block(reverse_size, reverse_size, marge_size, marge_size).transpose());
            const TVec<Scalar> &&br = reverse_bias_.head(reverse_size);
            const TVec<Scalar> &&bm = reverse_bias_.tail(marge_size);

            ComputePriorBySchurComplement(Hrr, Hrm, Hmr, Hmm, br, bm);
            break;
        }

        // [ Hmm Hmr ] [ bm ]
        // [ Hrm Hrr ] [ br ]
        case SortMargedVerticesDirection::kSortAtFront:
        default: {
            const TMat<Scalar> &&Hrr = reverse_hessian_.block(marge_size, marge_size, reverse_size, reverse_size);
            const TMat<Scalar> &&Hrm = reverse_hessian_.block(marge_size, 0, reverse_size, marge_size);
            const TMat<Scalar> &&Hmr = reverse_hessian_.block(0, marge_size, marge_size, reverse_size);
            const TMat<Scalar> Hmm = 0.5 * (reverse_hessian_.block(0, 0, marge_size, marge_size) +
                reverse_hessian_.block(0, 0, marge_size, marge_size).transpose());
            const TVec<Scalar> &&br = reverse_bias_.tail(reverse_size);
            const TVec<Scalar> &&bm = reverse_bias_.head(marge_size);

            ComputePriorBySchurComplement(Hrr, Hrm, Hmr, Hmm, br, bm);
            break;
        }
    }
}

// Create prior information, but only hessian and bias.
template <typename Scalar>
void Marginalizor<Scalar>::CreatePriorInformationOnlyHessianAndBias(const TMat<Scalar> &hessian,
                                                                    const TVec<Scalar> &bias,
                                                                    int32_t reverse_size,
                                                                    int32_t marge_size) {
    switch (options_.kSortDirection) {
        // [ Hrr Hrm ] [ br ]
        // [ Hmr Hmm ] [ bm ]
        case SortMargedVerticesDirection::kSortAtBack: {
            const TMat<Scalar> &&Hrr = hessian.block(0, 0, reverse_size, reverse_size);
            const TMat<Scalar> &&Hrm = hessian.block(0, reverse_size, reverse_size, marge_size);
            const TMat<Scalar> &&Hmr = hessian.block(reverse_size, 0, marge_size, reverse_size);
            const TMat<Scalar> Hmm = 0.5 * (hessian.block(reverse_size, reverse_size, marge_size, marge_size) +
                hessian.block(reverse_size, reverse_size, marge_size, marge_size).transpose());
            const TVec<Scalar> &&br = bias.head(reverse_size);
            const TVec<Scalar> &&bm = bias.tail(marge_size);

            SchurComplement(Hrr, Hrm, Hmr, Hmm, br, bm, reverse_hessian_, reverse_bias_);
            break;
        }

        // [ Hmm Hmr ] [ bm ]
        // [ Hrm Hrr ] [ br ]
        case SortMargedVerticesDirection::kSortAtFront:
        default: {
            const TMat<Scalar> &&Hrr = hessian.block(marge_size, marge_size, reverse_size, reverse_size);
            const TMat<Scalar> &&Hrm = hessian.block(marge_size, 0, reverse_size, marge_size);
            const TMat<Scalar> &&Hmr = hessian.block(0, marge_size, marge_size, reverse_size);
            const TMat<Scalar> Hmm = 0.5 * (hessian.block(0, 0, marge_size, marge_size) +
                hessian.block(0, 0, marge_size, marge_size).transpose());
            const TVec<Scalar> &&br = bias.tail(reverse_size);
            const TVec<Scalar> &&bm = bias.head(marge_size);

            SchurComplement(Hrr, Hrm, Hmr, Hmm, br, bm, reverse_hessian_, reverse_bias_);
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
    const TVec<Scalar> S = TVec<Scalar>((saes.eigenvalues().array() > kZeroFloat).select(saes.eigenvalues().array(), 0));
    const TVec<Scalar> S_inv = TVec<Scalar>((saes.eigenvalues().array() > kZeroFloat).select(saes.eigenvalues().array().inverse(), 0));
    const TVec<Scalar> S_sqrt = S.cwiseSqrt();
    const TVec<Scalar> S_inv_sqrt = S_inv.cwiseSqrt();

    // Calculate prior information, store them in graph problem.
    const TMat<Scalar> eigen_vectors = saes.eigenvectors().transpose();
    jacobian_t_inv = S_inv_sqrt.asDiagonal() * eigen_vectors;
    residual = - jacobian_t_inv * bias;
    jacobian = S_sqrt.asDiagonal() * eigen_vectors;
    hessian = jacobian.transpose() * jacobian;
    const TMat<Scalar> tmp_h = TMat<Scalar>((hessian.array().abs() > kZeroFloat).select(hessian.array(), 0));
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
    RETURN_FALSE_IF(row_index + dimension > hessian.rows());

    const int32_t size = static_cast<int32_t>(hessian.rows());
    const int32_t idx = static_cast<int32_t>(row_index);
    const int32_t dim = static_cast<int32_t>(dimension);

    switch (options_.kSortDirection) {
        // [ Hrr Hrm ] [ br ]
        // [ Hmr Hmm ] [ bm ]
        case SortMargedVerticesDirection::kSortAtBack: {
            // Move rows to be discarded to the bottom of hessian matrix.
            const TMat<Scalar> temp_rows = hessian.block(idx, 0, dim, size);
            const TMat<Scalar> temp_bottom_rows = hessian.block(idx + dim, 0, size - dim - idx, size);
            hessian.block(size - dim, 0, temp_rows.rows(), temp_rows.cols()).noalias() = temp_rows;
            hessian.block(idx, 0, temp_bottom_rows.rows(), temp_bottom_rows.cols()).noalias() = temp_bottom_rows;
            // Move rows to be discarded to the right of hessian matrix.
            const TMat<Scalar> temp_cols = hessian.block(0, idx, size, dim);
            const TMat<Scalar> temp_right_cols = hessian.block(0, idx + dim, size, size - dim - idx);
            hessian.block(0, size - dim, temp_cols.rows(), temp_cols.cols()).noalias() = temp_cols;
            hessian.block(0, idx, temp_right_cols.rows(), temp_right_cols.cols()).noalias() = temp_right_cols;
            // Move rows to be discarded to the bottom of bias vector.
            const TVec<Scalar> temp_bias = bias.segment(idx, dim);
            const TVec<Scalar> temp_bias_tail = bias.segment(idx + dim, size - dim - idx);
            bias.segment(size - dim, temp_bias.rows()).noalias() = temp_bias;
            bias.segment(idx, temp_bias_tail.rows()).noalias() = temp_bias_tail;

            break;
        }

        // [ Hmm Hmr ] [ bm ]
        // [ Hrm Hrr ] [ br ]
        case SortMargedVerticesDirection::kSortAtFront:
        default: {
            // Move rows to be discarded to the top of hessian matrix.
            const TMat<Scalar> temp_rows = hessian.block(idx, 0, dim, size);
            const TMat<Scalar> temp_top_rows = hessian.block(0, 0, idx, size);
            hessian.block(dim, 0, temp_top_rows.rows(), temp_top_rows.cols()).noalias() = temp_top_rows;
            hessian.block(0, 0, temp_rows.rows(), temp_rows.cols()).noalias() = temp_rows;
            // Move rows to be discarded to the left of hessian matrix.
            const TMat<Scalar> temp_cols = hessian.block(0, idx, size, dim);
            const TMat<Scalar> temp_left_cols = hessian.block(0, 0, size, idx);
            hessian.block(0, dim, temp_left_cols.rows(), temp_left_cols.cols()).noalias() = temp_left_cols;
            hessian.block(0, 0, temp_cols.rows(), temp_cols.cols()).noalias() = temp_cols;
            // Move rows to be discarded to the top of bias vector.
            const TVec<Scalar> temp_bias = bias.segment(idx, dim);
            const TVec<Scalar> temp_bias_head = bias.segment(0, idx);
            bias.segment(dim, temp_bias_head.rows()).noalias() = temp_bias_head;
            bias.segment(0, temp_bias.rows()).noalias() = temp_bias;

            break;
        }
    }

    return true;
}

}
