#include "solver_gn.h"
#include "slam_log_reporter.h"

namespace slam_solver {

/* Specialized Template Class Declaration. */
template class SolverGn<float>;
template class SolverGn<double>;

/* Class Gauss-Newton Solver Definition. */
template <typename Scalar>
void SolverGn<Scalar>::InitializeSolver() {
    step_scale_ = static_cast<Scalar>(1);
    // GN's linear system (H*dx = b) benefits from a direct solver for well-conditioned convergence.
    this->options().kLinearFunctionSolverType = Solver<Scalar>::LinearFunctionSolverType::kCholeskySolver;
    this->cost_at_linearized_point() = this->cost_at_latest_step();

    // Report information of initialize state if enabled.
    if (this->options().kEnableReportEachIteration) {
        ReportInfo("[GN] Init step scale is " << step_scale_ << ", cost is " << this->cost_at_latest_step() << "/" << this->cost_at_linearized_point() << "("
                                              << this->problem()->prior_residual().squaredNorm() << "), dx_norm is " << this->dx().norm());
    }
}

template <typename Scalar>
void SolverGn<Scalar>::SolveIncrementalFunction() {
    auto &hessian = this->problem()->hessian();
    auto &bias = this->problem()->bias();
    const int32_t hessian_size = hessian.rows();

    if (hessian_size == 0) {
        this->dx().setZero(0);
        return;
    }

    // Solve Gauss-Newton incremental function: H * dx_gn = bias
    const int32_t reverse = this->problem()->full_size_of_dense_vertices();
    const int32_t marg = this->problem()->full_size_of_sparse_vertices();

    if (marg == 0) {
        // Directly solve the incremental function.
        this->SolveLinearlizedFunction(hessian, bias, dx_gn_);
    } else {
        dx_gn_.resize(hessian_size);

        // Firstly solve dense parameters.
        this->problem()->MarginalizeSparseVerticesInHessianAndBias(reverse_hessian_, reverse_bias_);
        reverse_dx_.resize(reverse);
        this->SolveLinearlizedFunction(reverse_hessian_, reverse_bias_, reverse_dx_);
        dx_gn_.head(reverse) = reverse_dx_;

        // Secondly solve sparse parameters.
        marg_bias_ = bias.tail(marg) - hessian.block(reverse, 0, marg, reverse) * reverse_dx_;
        for (const auto &vertex: this->problem()->sparse_vertices()) {
            const int32_t index = vertex->ColIndex();
            const int32_t dim = vertex->GetIncrementDimension();
            this->SolveLinearlizedFunction(hessian.block(index, index, dim, dim), marg_bias_.segment(index - reverse, dim), marg_dx_);
            dx_gn_.segment(index, dim) = marg_dx_;
        }
    }

    // Apply step scale (for backtracking line search).
    this->dx() = step_scale_ * dx_gn_;
}

// Check if one update step is valid.
template <typename Scalar>
bool SolverGn<Scalar>::IsUpdateValid(Scalar min_allowed_gain_rate) {
    const Scalar cost_ratio = this->cost_at_linearized_point() - this->cost_at_latest_step();

    // Report information of this iteration if enabled.
    const Scalar delta_x_norm = this->dx().norm();
    if (this->options().kEnableReportEachIteration) {
        ReportInfo("[GN] step_scale is " << step_scale_ << ", cost_reduction is " << cost_ratio << ", cost is " << this->cost_at_latest_step() << "/"
                                         << this->cost_at_linearized_point() << "(" << this->problem()->prior_residual().squaredNorm() << "), dx_norm is "
                                         << delta_x_norm);
    }

    if (cost_ratio > min_allowed_gain_rate && std::isfinite(this->cost_at_latest_step())) {
        // Cost decreased: accept step, reset step scale for next iteration.
        step_scale_ = static_cast<Scalar>(1);
        this->cost_at_linearized_point() = this->cost_at_latest_step();
        return true;
    } else {
        // Cost increased: reduce step scale via backtracking, reject step.
        step_scale_ *= sub_options_.kStepScaleDecreaseFactor;
        step_scale_ = std::max(step_scale_, sub_options_.kMinStepScale);
        return false;
    }
}

}  // namespace slam_solver
