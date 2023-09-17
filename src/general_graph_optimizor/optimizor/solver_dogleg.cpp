#include "solver_dogleg.h"

namespace SLAM_SOLVER {

/* Specialized Template Class Declaration. */
template class SolverDogleg<float>;
template class SolverDogleg<double>;

/* Class Dogleg Solver Definition. */
template <typename Scalar>
void SolverDogleg<Scalar>::InitializeSolver() {
    // Use 'this->' to run function definded in basis class.
    radius_ = dogleg_options_.kInitRadius;
    this->cost_at_linearized_point() = this->cost_at_latest_step();

    auto &hessian = this->problem()->hessian();
    radius_ = std::max(radius_, hessian.diagonal().array().maxCoeff());
    radius_ = std::min(radius_, dogleg_options_.kMaxRadius);
}

template <typename Scalar>
void SolverDogleg<Scalar>::SolveIncrementalFunction() {
    auto &hessian = this->problem()->hessian();
    auto &bias = this->problem()->bias();
    const int32_t hessian_size = hessian.rows();

    // Compute Gradient descent method incremental parameters.
    alpha_ = bias.squaredNorm() / (bias.dot(hessian * bias));
    dx_sd_ = alpha_ * bias;

    // Solve Gauss newton method incremental function.
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
        for (const auto &vertex : this->problem()->sparse_vertices()) {
            const int32_t index = vertex->ColIndex();
            const int32_t dim = vertex->GetIncrementDimension();
            this->SolveLinearlizedFunction(hessian.block(index, index, dim, dim),
                                           marg_bias_.segment(index - reverse, dim),
                                           marg_dx_);
            dx_gn_.segment(index, dim) = marg_dx_;
        }
    }

    // Combine two incremantal parameters with trust radius.
    const Scalar norm_sd = dx_sd_.norm();
    const Scalar norm_gn = dx_gn_.norm();
    if (std::isnan(norm_gn)) {
        const Scalar scale = radius_ > norm_sd ? static_cast<Scalar>(1) : radius_ / norm_sd;
        this->dx() = dx_sd_ * scale;
    } else {
        if (norm_gn <= radius_ && norm_sd <= radius_) {
            this->dx() = dx_gn_;
        } else if (norm_gn >= radius_ && norm_sd >= radius_) {
            this->dx() = dx_sd_ * radius_ / norm_sd;
        } else {
            diff_dx_sd_gn_ = dx_gn_ - dx_sd_;
            const Scalar a = diff_dx_sd_gn_.dot(diff_dx_sd_gn_);
            const Scalar b = static_cast<Scalar>(2) * dx_sd_.dot(diff_dx_sd_gn_);
            const Scalar c = norm_sd * norm_sd - radius_ * radius_;
            const Scalar w = (std::sqrt(b * b - static_cast<Scalar>(4) * a * c) - b) / (static_cast<Scalar>(2) * a);
            this->dx() = dx_sd_ + w * diff_dx_sd_gn_;
        }
    }
}

// Check if one update step is valid.
template <typename Scalar>
bool SolverDogleg<Scalar>::IsUpdateValid(Scalar min_allowed_gain_rate) {
    // Reference: The Levenberg-Marquardt method for nonlinear least squares curve-fitting problems.pdf
    auto &hessian = this->problem()->hessian();
    auto &bias = this->problem()->bias();
    const Scalar scale = static_cast<Scalar>(2) * bias.dot(this->dx()) + this->dx().dot(hessian * this->dx()) + static_cast<Scalar>(1e-6);
    const Scalar rho = static_cast<Scalar>(0.5) * (this->cost_at_linearized_point() - this->cost_at_latest_step()) / scale;

    // Report information of this iteration if enabled.
    const Scalar delta_x_norm = this->dx().norm();
    if (this->options().kEnableReportEachIteration) {
        ReportInfo("[Dogleg] radius is " << radius_ << ", rho is " << rho << ", cost is " << this->cost_at_latest_step() << "/" <<
            this->cost_at_linearized_point() << ", dx_norm is " << delta_x_norm);
    }

    if (rho > min_allowed_gain_rate && std::isfinite(this->cost_at_latest_step()) && !std::isnan(this->cost_at_latest_step())) {
        if (rho > static_cast<Scalar>(0.75)) {
            radius_ = std::max(radius_, static_cast<Scalar>(3) * delta_x_norm);
        } else if (rho < static_cast<Scalar>(0.25)) {
            radius_ *= static_cast<Scalar>(0.5);
        }
        this->cost_at_linearized_point() = this->cost_at_latest_step();
        return true;
    } else {
        radius_ *= static_cast<Scalar>(0.25);
        return false;
    }
}

}
