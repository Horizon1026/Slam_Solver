#ifndef _GENERAL_GRAPH_OPTIMIZOR_SOLVER_H_
#define _GENERAL_GRAPH_OPTIMIZOR_SOLVER_H_

#include "basic_type.h"
#include "graph.h"

namespace slam_solver {

/* Class Solver Declaration. */
template <typename Scalar>
class Solver {

public:
    enum class LinearFunctionSolverType {
        kCholeskySolver = 0,
        kPcgSolver = 1,
        kQrSolver = 2,
        kSvdSolver = 3,
    };

    struct Options {
        int32_t kMaxIteration = 30;
        Scalar kMaxConvergedSquaredStepLength = 1e-6;
        Scalar kMaxPcgSolverCostDecreaseRate = 1e-6;
        Scalar kMaxPcgSolverConvergedResidual = 1e-6;
        bool kEnableReportEachIteration = true;
        bool kOnlyUseFirstEstimatedJacobian = false;
        bool kEnableDegenerateElimination = true;
        Scalar kMinEigenValueThresholdForDegenerateElimination = 1e-6;
        float kMaxCostTimeInSecond = 1.0f;
        LinearFunctionSolverType kLinearFunctionSolverType = LinearFunctionSolverType::kPcgSolver;
        bool kEnableJacobianScaling = true;
    };

public:
    Solver() = default;
    virtual ~Solver() = default;

    // Solve graph optimization problem.
    bool Solve(bool use_prior = false);

    // Initialize solver init value with first incremental function and so on.
    virtual void InitializeSolver() = 0;

    // Construct incremental function, Hx = b or Jx = -r.
    // Hx = b is default.
    virtual void ConstructIncrementalFunction(bool use_prior = false);

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
    void SolveLinearlizedFunction(const TMat<Scalar> &A, const TVec<Scalar> &b, TVec<Scalar> &x);

    // Reference for member variables.
    Options &options() { return options_; }
    Graph<Scalar> *&problem() { return problem_; }

    Scalar &cost_at_linearized_point() { return cost_at_linearized_point_; }
    Scalar &cost_at_latest_step() { return cost_at_latest_step_; }

    TVec<Scalar> &dx() { return dx_; }
    TVec<Scalar> &prior_bias_backup() { return prior_bias_backup_; }
    TVec<Scalar> &prior_residual_backup() { return prior_residual_backup_; }

    // Const reference for member variables.
    const Options &options() const { return options_; }
    const Graph<Scalar> *problem() const { return problem_; }

    const Scalar &cost_at_linearized_point() const { return cost_at_linearized_point_; }
    const Scalar &cost_at_latest_step() const { return cost_at_latest_step_; }

    const TVec<Scalar> &dx() const { return dx_; }
    const TVec<Scalar> &prior_bias_backup() const { return prior_bias_backup_; }
    const TVec<Scalar> &prior_residual_backup() const { return prior_residual_backup_; }

private:
    // Solve linear system with specified solver.
    void SolveLinearSystem(const TMat<Scalar> &A, const TVec<Scalar> &b, TVec<Scalar> &x);
    // Use PCG solver to solve linear system.
    void SolvePcg(const TMat<Scalar> &A, const TVec<Scalar> &b, TVec<Scalar> &x);
    // Eliminate degeneracy of the linear system.
    void EliminateDegeneracy(const TMat<Scalar> &A, TVec<Scalar> &x);

private:
    // General options for solver.
    Options options_;

    // The graph optimization problem to be solved.
    Graph<Scalar> *problem_ = nullptr;

    // The summary of residual at linearized point.
    Scalar cost_at_linearized_point_ = 0;
    Scalar cost_at_latest_step_ = 0;

    // The increment of all parameters.
    TVec<Scalar> dx_;

    // Backup for prior information.
    TVec<Scalar> prior_bias_backup_ = TVec3<Scalar>::Zero();
    TVec<Scalar> prior_residual_backup_ = TVec3<Scalar>::Zero();
};

}  // namespace slam_solver

#endif  // end of _GENERAL_GRAPH_OPTIMIZOR_SOLVER_H_
