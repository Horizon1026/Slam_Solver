#include "solver_gd.h"
#include "slam_log_reporter.h"

namespace slam_solver {

/* Specialized Template Class Declaration. */
template class SolverGd<float>;
template class SolverGd<double>;

/* Class Gradient Descent Solver Definition. */
template <typename Scalar>
void SolverGd<Scalar>::InitializeSolver() {
    learning_rate_ = sub_options_.kInitLearningRate;
    this->cost_at_linearized_point() = this->cost_at_latest_step();

    // Report information of initialize state if enabled.
    if (this->options().kEnableReportEachIteration) {
        ReportInfo("[GD] Init learning rate is " << learning_rate_ << ", cost is " << this->cost_at_latest_step() << "/" << this->cost_at_linearized_point()
                                                 << "(" << this->problem()->prior_residual().squaredNorm() << "), dx_norm is " << this->dx().norm());
    }
}

template <typename Scalar>
void SolverGd<Scalar>::ConstructIncrementalFunction(bool use_prior) {
    // Override the base class to construct the full Jacobian and residual
    // instead of the Hessian and bias. This gives us direct access to J and r
    // for computing the gradient g = J^T * r.
    this->problem()->ConstructFullSizeJacobianAndResidual(use_prior);
}

template <typename Scalar>
void SolverGd<Scalar>::SolveIncrementalFunction() {
    auto &jacobian = this->problem()->jacobian();
    auto &residual = this->problem()->residual();
    const int32_t param_size = jacobian.cols();

    if (param_size == 0) {
        this->dx().setZero(0);
        return;
    }

    // Compute gradient directly from Jacobian and residual: g = J^T * r.
    // This avoids constructing the full Hessian (J^T * J).
    const TVec<Scalar> gradient = jacobian.transpose() * residual;

    // Compute column norms of Jacobian for diagonal preconditioning.
    // diag_JtJ_i = ||J(:,i)||^2 = (J^T * J)_{ii}
    const TVec<Scalar> diag_JtJ = jacobian.colwise().squaredNorm();

    // Gradient descent step with diagonal scaling:
    // The gradient g = J^T * r is the direction of steepest ascent.
    // For descent: dx = -learning_rate * g_i / ||J(:,i)||^2
    // In the H&b convention: bias = -g, so dx = learning_rate * bias_i / H_ii
    this->dx().resize(param_size);
    for (int32_t i = 0; i < param_size; ++i) {
        if (diag_JtJ(i) > std::numeric_limits<Scalar>::epsilon()) {
            this->dx()(i) = -learning_rate_ * gradient(i) / diag_JtJ(i);
        } else {
            this->dx()(i) = -learning_rate_ * gradient(i);
        }
    }
}

// Check if one update step is valid.
template <typename Scalar>
bool SolverGd<Scalar>::IsUpdateValid(Scalar min_allowed_gain_rate) {
    const Scalar cost_ratio = this->cost_at_linearized_point() - this->cost_at_latest_step();

    // Report information of this iteration if enabled.
    if (this->options().kEnableReportEachIteration) {
        ReportInfo("[GD] learning rate is " << learning_rate_ << ", cost_reduction is " << cost_ratio << ", cost is " << this->cost_at_latest_step() << "/"
                                            << this->cost_at_linearized_point() << "(" << this->problem()->prior_residual().squaredNorm() << "), dx_norm is "
                                            << this->dx().norm());
    }

    if (cost_ratio > min_allowed_gain_rate && std::isfinite(this->cost_at_latest_step())) {
        // Cost decreased: accept step and slightly increase learning rate (speed up convergence).
        learning_rate_ *= sub_options_.kLearningRateIncreaseFactor;
        learning_rate_ = std::min(learning_rate_, sub_options_.kMaxLearningRate);
        this->cost_at_linearized_point() = this->cost_at_latest_step();
        return true;
    } else {
        // Cost increased: reduce learning rate and reject step.
        learning_rate_ *= sub_options_.kLearningRateDecreaseFactor;
        learning_rate_ = std::max(learning_rate_, sub_options_.kMinLearningRate);
        return false;
    }
}

}  // namespace slam_solver
