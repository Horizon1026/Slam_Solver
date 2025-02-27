#include "solver.h"
#include "tick_tock.h"
#include "slam_log_reporter.h"
#include "slam_basic_math.h"

namespace SLAM_SOLVER {

/* Specialized Template Class Declaration. */
template class Solver<float>;
template class Solver<double>;

/* Class Solver Definition. */
template <typename Scalar>
bool Solver<Scalar>::Solve(bool use_prior) {
    if (problem_ == nullptr) {
        return false;
    }
    float time_cost = 0.0f;
    SLAM_UTILITY::TickTock timer;
    timer.TockTickInSecond();

    // Sort all vertices, determine their location in incremental function.
    problem_->SortVertices(false);
    // Linearize the non-linear problem, construct incremental function.
    cost_at_latest_step_ = problem_->ComputeResidualForAllEdges(use_prior);
    problem_->ComputeJacobiansForAllEdges();
    ConstructIncrementalFunction(use_prior);
    // Initialize solver.
    InitializeSolver();

    for (int32_t iter = 0; iter < options_.kMaxIteration; ++iter) {
        // Solve function to get increment of all parameters.
        SolveIncrementalFunction();
        // Update all parameters.
        UpdateParameters(use_prior);
        // Recompute residual after update.
        cost_at_latest_step_ = problem_->ComputeResidualForAllEdges(use_prior);
        // If this step is valid, perpare for next iteration.
        if (IsUpdateValid()) {
            cost_at_linearized_point_ = cost_at_latest_step_;
            if (!options_.kOnlyUseFirstEstimatedJacobian) {
                problem_->ComputeJacobiansForAllEdges();
            }
            ConstructIncrementalFunction(use_prior);
        } else {
            RollBackParameters(use_prior);
        }

        // If converged, the iteration can be stopped now.
        if (IsConvergedAfterUpdate(iter)) {
            break;
        }
        // If not converged, check for timeout.
        const float time_cost_this_step = timer.TockInSecond() - time_cost;
        time_cost = timer.TockInSecond();
        if (time_cost + time_cost_this_step > options_.kMaxCostTimeInSecond) {
            break;
        }

    }

    return true;
}

// Construct incremental function, Hx = b or Jx = -r.
template <typename Scalar>
void Solver<Scalar>::ConstructIncrementalFunction(bool use_prior) {
    problem_->ConstructFullSizeHessianAndBias(use_prior);
}

// Update or rollback all vertices and prior.
template <typename Scalar>
void Solver<Scalar>::UpdateParameters(bool use_prior) {
    // Update and backup all vertices.
    problem_->UpdateAllVertices(dx_);

    // Update and backup prior information.
    if (use_prior && problem_->prior_hessian().rows() > 0) {
        prior_bias_backup_ = problem_->prior_bias();
        prior_residual_backup_ = problem_->prior_residual();
        problem_->prior_bias() -= problem_->prior_hessian() * dx_.head(problem_->prior_hessian().cols());
        problem_->prior_residual() = - problem_->prior_jacobian_t_inv() * problem_->prior_bias();
    }
}

template <typename Scalar>
void Solver<Scalar>::RollBackParameters(bool use_prior) {
    // Roll back all vertices.
    problem_->RollBackAllVertices();

    // Roll back prior information.
    if (use_prior && problem_->prior_hessian().rows() > 0) {
        problem_->prior_bias() = prior_bias_backup_;
        problem_->prior_residual() = prior_residual_backup_;
    }
}

// Check if the iteration converged.
template <typename Scalar>
bool Solver<Scalar>::IsConvergedAfterUpdate(int32_t iter) {
    if (Eigen::isnan(dx_.array()).any()) {
        ReportError("[Solver] Incremental param is nan. Not converged.");
        return false;
    }

    if (dx_.squaredNorm() < options_.kMaxConvergedSquaredStepLength) {
        return true;
    }

    return false;
}

// Default Use PCG solver to solve linearlized function. Other solvers can also be added here.
template <typename Scalar>
void Solver<Scalar>::SolveLinearlizedFunction(const TMat<Scalar> &A,
                                              const TVec<Scalar> &b,
                                              TVec<Scalar> &x) {
    switch (options_.kLinearFunctionSolverType) {
        case LinearFunctionSolverType::kPcgSolver: {
            const int32_t size = b.rows();
            const int32_t maxIteration = size;

            // If initial value is ok, return.
            x.setZero(size);
            TVec<Scalar> r0(b);  // initial r = b - A*0 = b
            if (r0.norm() < options_.kMaxPcgSolverConvergedResidual) {
                return;
            }

            // Compute precondition matrix.
            TVec<Scalar> M_inv_diag = A.diagonal();
            M_inv_diag.array() = static_cast<Scalar>(1) / M_inv_diag.array();
            for (int32_t i = 0; i < M_inv_diag.rows(); ++i) {
                if (std::isinf(M_inv_diag(i))) {
                    M_inv_diag(i) = 0;
                }
            }
            TVec<Scalar> z0 = M_inv_diag.array() * r0.array();    // solve M * z0 = r0

            // Get first basis vector, compute weight alpha, update x.
            TVec<Scalar> p(z0);
            TVec<Scalar> w = A * p;
            Scalar r0z0 = r0.dot(z0);
            Scalar alpha = r0z0 / p.dot(w);
            x += alpha * p;
            TVec<Scalar> r1 = r0 - alpha * w;

            // Set threshold to check if converged.
            const Scalar threshold = options_.kMaxPcgSolverCostDecreaseRate * r0.norm();

            int32_t i = 0;
            TVec<Scalar> z1;
            while (r1.norm() > threshold && i < maxIteration) {
                i++;
                z1 = M_inv_diag.array() * r1.array();
                const Scalar r1z1 = r1.dot(z1);
                const Scalar belta = r1z1 / r0z0;
                z0 = z1;
                r0z0 = r1z1;
                r0 = r1;
                p = belta * p + z1;
                w = A * p;
                alpha = r1z1 / p.dot(w);
                x += alpha * p;
                r1 -= alpha * w;
            }
            break;
        }

        case LinearFunctionSolverType::kCholeskySolver: {
            x = A.template ldlt().solve(b);
            break;
        }
        case LinearFunctionSolverType::kQrSolver: {
            x = A.template colPivHouseholderQr().solve(b);
            break;
        }
        case LinearFunctionSolverType::kSvdSolver: {
            x = A.template jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b);
            break;
        }
        default: {
            ReportError("[Solver] Linear function solver type not supported.");
            break;
        }
    }

    if (!options_.kEnableDegenerateElimination) {
        return;
    }

    // Reference: On Degeneracy of Optimization-based State Estimation Problems.pdf (There is typo in this paper.)
    // For Hx = b, which can be written as J.transpose() * J * dx = -J.transpose() * r. Actually we solve J * dx = -r.
    // Compute eigen value and vector of A.
    Eigen::SelfAdjointEigenSolver<TMat<Scalar>> solver(A);
    if (solver.info() != Eigen::Success) {
        return;
    }
    // Eigen values are sorted in increasing order. The smallest eigen value is at the first position.
    const TVec<Scalar> &eigen_values = solver.eigenvalues().real();
    if (eigen_values(0) > options_.kMinEigenValueThresholdForDegenerateElimination) {
        return;
    }
    // Eigen vectors are sorted by eigen values as columns.
    const TMat<Scalar> &eigen_vectors = solver.eigenvectors().real();
    const TMat<Scalar> &matrix_f_inv = eigen_vectors;   // For eigen vectors matrix, transpose is equal to inverse.
    // Matrix f and matrix u are both rows of eigen vectors. So the inverse of them are cols of eigen vectors.
    // Select the directions of full-conditioned subspace.
    TMat<Scalar> matrix_u_inv = eigen_vectors;
    for (int32_t i = 0; i < matrix_u_inv.cols(); ++i) {
        if (eigen_values(i) < options_.kMinEigenValueThresholdForDegenerateElimination) {
            matrix_u_inv.col(i).setZero();
        }
    }
    // Compute the result projected into full-conditioned subspace.
    x = matrix_f_inv * matrix_u_inv.transpose() * x;

}

}
