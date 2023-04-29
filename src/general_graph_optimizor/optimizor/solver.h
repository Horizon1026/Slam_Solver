#ifndef _GENERAL_GRAPH_OPTIMIZOR_SOLVER_H_
#define _GENERAL_GRAPH_OPTIMIZOR_SOLVER_H_

#include "datatype_basic.h"
#include "graph.h"

namespace SLAM_SOLVER {

template <typename Scalar>
struct SolverOptions {
    int32_t kMaxIteration = 20;
    Scalar kMaxConvergedSquaredStepLength = 1e-7;
};

/* Class Solver Declaration. */
template <typename Scalar>
class Solver {

public:
    Solver() = default;
    virtual ~Solver() = default;

    // Solve graph optimization problem.
    bool Solve();

    // Initialize solver init value with first incremental function and so on.
    virtual void InitializeSolver() = 0;

    // Solve Hx=b, or Jx=-r, in order to get delta parameters.
    virtual void SolveIncrementalFunction() = 0;

    // Update or rollback all vertices and prior.
    void UpdateParameters(bool use_prior = false);
    void RollBackParameters(bool use_prior = false);

    // Check if one update step is valid.
    virtual bool IsUpdateValid(Scalar min_allowed_gain_rate = 0) = 0;

    // Check if the iteration converged.
    bool IsConvergedAfterUpdate(int32_t iter);

    // Use PCG solver to solve linearlized function.
    void SolveLinearlizedFunction(const TMat<Scalar> &A,
                                  const TVec<Scalar> &b,
                                  TVec<Scalar> &x);

    // Reference for member varibles.
    SolverOptions<Scalar> &options() { return options_; }
    Graph<Scalar> &problem() { return problem_; }

    Scalar &cost_at_linearized_point() { return cost_at_linearized_point_; }
    Scalar &cost_at_latest_step() { return cost_at_latest_step_; }

    TVec<Scalar> &dx() { return dx_; }
    TVec<Scalar> &prior_bias_backup() { return prior_bias_backup_; }
    TVec<Scalar> &prior_residual_backup() { return prior_residual_backup_; }

private:
    // General options for solver.
    SolverOptions<Scalar> options_;

    // The graph optimization problem to be solved.
    Graph<Scalar> problem_;

    // The summary of residual at linearized point.
    Scalar cost_at_linearized_point_ = 0;
    Scalar cost_at_latest_step_ = 0;

    // The increment of all parameters.
    TVec<Scalar> dx_;

    // Backup for prior information.
    TVec<Scalar> prior_bias_backup_ = TVec3<Scalar>::Zero();
    TVec<Scalar> prior_residual_backup_ = TVec3<Scalar>::Zero();

};

/* Class Solver Definition. */
template <typename Scalar>
bool Solver<Scalar>::Solve() {
    // Linearize the non-linear problem, construct incremental function.
    cost_at_latest_step_ = problem_.ComputeResidualForAllEdges();
    problem_.ComputeJacobiansForAllEdges();
    problem_.ConstructFullSizeHessianAndBias(true);

    // Initialize solver.
    InitializeSolver();

    for (int32_t iter = 0; iter < options_.kMaxIteration; ++iter) {
        // Solve function to get increment of all parameters.
        SolveIncrementalFunction();

        // Update all parameters.
        UpdateParameters();

        // Recompute residual after update.
        cost_at_latest_step_ = problem_.ComputeResidualForAllEdges();

        // If converged, the iteration can be stopped now.
        if (IsConvergedAfterUpdate(iter)) {
            break;
        }

        // If this step is valid, perpare for next iteration.
        if (IsUpdateValid()) {
            problem_.ComputeJacobiansForAllEdges();
            problem_.ConstructFullSizeHessianAndBias(true);
        } else {
            RollBackParameters();
        }
    }

    return true;
}

// Update or rollback all vertices and prior.
template <typename Scalar>
void Solver<Scalar>::UpdateParameters(bool use_prior) {
    // Update and backup all vertices.
    problem_.UpdateAllVertices(dx_);

    // Update and backup prior information.
    if (use_prior && problem_.prior_hessian().size() > 0) {
        prior_bias_backup_ = problem_.prior_bias();
        prior_residual_backup_ = problem_.prior_residual();
        problem_.prior_bias() -= problem_.prior_hessian() * dx_.head(problem_.prior_hessian().cols());
        problem_.prior_residual() = - problem_.prior_jacobian_t_inv() * problem_.prior_bias();
    }
}

template <typename Scalar>
void Solver<Scalar>::RollBackParameters(bool use_prior) {
    // Roll back all vertices.
    problem_.RollBackAllVertices();

    // Roll back prior information.
    if (use_prior && problem_.prior_hessian().size() > 0) {
        problem_.prior_bias() = prior_bias_backup_;
        problem_.prior_residual() = prior_residual_backup_;
    }
}

// Check if the iteration converged.
template <typename Scalar>
bool Solver<Scalar>::IsConvergedAfterUpdate(int32_t iter) {
    if (dx_.squaredNorm() < options_.kMaxConvergedSquaredStepLength) {
        return true;
    }

    return false;
}

// Use PCG solver to solve linearlized function.
template <typename Scalar>
void Solver<Scalar>::SolveLinearlizedFunction(const TMat<Scalar> &A,
                                              const TVec<Scalar> &b,
                                              TVec<Scalar> &x) {
    const int32_t size = b.rows();
    const int32_t maxIteration = size;

    // If initial value is ok, return.
    x.setZero(size);
    TVec<Scalar> r0(b);  // initial r = b - A*0 = b
    if (r0.norm() < static_cast<Scalar>(1e-6)) {
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
    const Scalar threshold = static_cast<Scalar>(1e-6) * r0.norm();

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
}

}

#endif // end of _GENERAL_GRAPH_OPTIMIZOR_SOLVER_H_
