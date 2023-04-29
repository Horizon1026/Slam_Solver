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
    void UpdateParameters();
    void RollBackParameters();

    // Check if one update step is valid.
    virtual bool IsUpdateValid(Scalar min_allowed_gain_rate = 0) = 0;

    // Check if the iteration converged.
    bool IsConvergedAfterUpdate(int32_t iter);

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
void Solver<Scalar>::UpdateParameters() {
    // Update and backup all vertices.
    problem_.UpdateAllVertices(dx_);

    // Update and backup prior information.
    prior_bias_backup_ = problem_.prior_bias();
    prior_residual_backup_ = problem_.prior_residual();
    problem_.prior_bias() -= problem_.prior_hessian() * dx_.head(problem_.prior_hessian().cols());
    problem_.prior_residual() = - problem_.prior_jacobian_t_inv() * problem_.prior_bias();
}

template <typename Scalar>
void Solver<Scalar>::RollBackParameters() {
    // Roll back all vertices.
    problem_.RollBackAllVertices();

    // Roll back prior information.
    problem_.prior_bias() = prior_bias_backup_;
    problem_.prior_residual() = prior_residual_backup_;
}

// Check if the iteration converged.
template <typename Scalar>
bool Solver<Scalar>::IsConvergedAfterUpdate(int32_t iter) {
    if (dx_.squaredNorm() < options_.kMaxConvergedSquaredStepLength) {
        return true;
    }

    return false;
}

}

#endif // end of _GENERAL_GRAPH_OPTIMIZOR_SOLVER_H_
