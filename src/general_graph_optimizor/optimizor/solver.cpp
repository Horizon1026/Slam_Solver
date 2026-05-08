#include "solver.h"
#include "slam_basic_math.h"
#include "slam_log_reporter.h"
#include "slam_operations.h"
#include "tick_tock.h"

namespace slam_solver {

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
    slam_utility::TickTock timer;
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
        problem_->prior_residual() = -problem_->prior_jacobian_t_inv() * problem_->prior_bias();
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
void Solver<Scalar>::SolveLinearlizedFunction(const TMat<Scalar> &A, const TVec<Scalar> &b, TVec<Scalar> &x) {
    const int32_t size = b.rows();
    x.setZero(size);
    RETURN_IF(size == 0 || A.rows() != size || A.cols() != size || !A.allFinite() || !b.allFinite());

    // Prepare for solving.
    const TMat<Scalar> *A_ptr = &A;
    const TVec<Scalar> *b_ptr = &b;

    // Step 1: Jacobian scaling.
    TVec<Scalar> D_inv_diag = TVec<Scalar>::Ones(size);
    TMat<Scalar> A_scaled;
    TVec<Scalar> b_scaled;
    if (options_.kEnableJacobianScaling) {
        for (int i = 0; i < size; ++i) {
            Scalar diagonal = A(i, i);
            if (diagonal > static_cast<Scalar>(0)) {
                D_inv_diag(i) = static_cast<Scalar>(1) / std::sqrt(diagonal);
            }
        }
        A_scaled = D_inv_diag.asDiagonal() * A * D_inv_diag.asDiagonal();
        b_scaled = D_inv_diag.array() * b.array();
        A_ptr = &A_scaled;
        b_ptr = &b_scaled;
    }

    // Step 2: Solve linear system.
    SolveLinearSystem(*A_ptr, *b_ptr, x);
    if (options_.kEnableJacobianScaling) {
        x = D_inv_diag.array() * x.array();
    }

    // Step 3: Eliminate degeneracy.
    EliminateDegeneracy(A, x);
}

template <typename Scalar>
void Solver<Scalar>::SolveLinearSystem(const TMat<Scalar> &A, const TVec<Scalar> &b, TVec<Scalar> &x) {
    switch (options_.kLinearFunctionSolverType) {
        case LinearFunctionSolverType::kPcgSolver: {
            SolvePcg(A, b, x);
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
}

template <typename Scalar>
void Solver<Scalar>::SolvePcg(const TMat<Scalar> &A, const TVec<Scalar> &b, TVec<Scalar> &x) {
    const int32_t size = b.rows();
    const int32_t maxIteration = size;

    // Initial r = b - A*0 = b.
    TVec<Scalar> r0(b);
    for (int32_t row = 0; row < size; ++row) {
        if (A.row(row).cwiseAbs().maxCoeff() == static_cast<Scalar>(0) && A.col(row).cwiseAbs().maxCoeff() == static_cast<Scalar>(0)) {
            r0(row) = static_cast<Scalar>(0);
        }
    }
    RETURN_IF(r0.norm() < options_.kMaxPcgSolverConvergedResidual);

    // Compute precondition matrix.
    TVec<Scalar> M_inv_diag = TVec<Scalar>::Zero(size);
    for (int32_t i = 0; i < size; ++i) {
        if (A(i, i) != static_cast<Scalar>(0) && std::isfinite(A(i, i))) {
            M_inv_diag(i) = static_cast<Scalar>(1) / A(i, i);
        }
    }
    TVec<Scalar> z0 = M_inv_diag.array() * r0.array();  // solve M * z0 = r0

    // Get first basis vector, compute weight alpha, update x.
    TVec<Scalar> p(z0);
    TVec<Scalar> w = A * p;
    Scalar r0z0 = r0.dot(z0);
    Scalar pAw = p.dot(w);
    RETURN_IF(r0z0 == static_cast<Scalar>(0) || pAw == static_cast<Scalar>(0) || !std::isfinite(r0z0) || !std::isfinite(pAw));
    Scalar alpha = r0z0 / pAw;
    RETURN_IF(!std::isfinite(alpha));
    x += alpha * p;
    TVec<Scalar> r1 = r0 - alpha * w;
    for (int32_t row = 0; row < size; ++row) {
        if (M_inv_diag(row) == static_cast<Scalar>(0)) {
            r1(row) = static_cast<Scalar>(0);
        }
    }

    // Set threshold to check if converged.
    const Scalar threshold = options_.kMaxPcgSolverCostDecreaseRate * r0.norm();

    int32_t i = 0;
    TVec<Scalar> z1;
    while (r1.norm() > threshold && i < maxIteration) {
        i++;
        z1 = M_inv_diag.array() * r1.array();
        const Scalar r1z1 = r1.dot(z1);
        BREAK_IF(r0z0 == static_cast<Scalar>(0) || r1z1 == static_cast<Scalar>(0) || !std::isfinite(r0z0) || !std::isfinite(r1z1));
        const Scalar belta = r1z1 / r0z0;
        BREAK_IF(!std::isfinite(belta));
        z0 = z1;
        r0z0 = r1z1;
        r0 = r1;
        p = belta * p + z1;
        w = A * p;
        pAw = p.dot(w);
        BREAK_IF(pAw == static_cast<Scalar>(0) || !std::isfinite(pAw));
        alpha = r1z1 / pAw;
        BREAK_IF(!std::isfinite(alpha));
        x += alpha * p;
        r1 -= alpha * w;
        for (int32_t row = 0; row < size; ++row) {
            if (M_inv_diag(row) == static_cast<Scalar>(0)) {
                r1(row) = static_cast<Scalar>(0);
            }
        }
    }

    if (!x.allFinite()) {
        x.setZero(size);
    }
}

template <typename Scalar>
void Solver<Scalar>::EliminateDegeneracy(const TMat<Scalar> &A, TVec<Scalar> &x) {
    RETURN_IF(!options_.kEnableDegenerateElimination);

    // Reference: On Degeneracy of Optimization-based State Estimation Problems.pdf (There is typo in this paper.)
    // For Hx = b, which can be written as J.transpose() * J * dx = -J.transpose() * r. Actually we solve J * dx = -r.
    // Compute eigen value and vector of A.
    Eigen::SelfAdjointEigenSolver<TMat<Scalar>> solver(A);
    RETURN_IF(solver.info() != Eigen::Success);
    // Eigen values are sorted in increasing order. The smallest eigen value is at the first position.
    const TVec<Scalar> &eigen_values = solver.eigenvalues().real();
    RETURN_IF(eigen_values(0) > options_.kMinEigenValueThresholdForDegenerateElimination);
    // Eigen vectors are sorted by eigen values as columns.
    const TMat<Scalar> &eigen_vectors = solver.eigenvectors().real();
    const TMat<Scalar> &matrix_f_inv = eigen_vectors;  // For eigen vectors matrix, transpose is equal to inverse.
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

}  // namespace slam_solver
