#ifndef _GENERAL_GRAPH_OPTIMIZOR_SOLVER_GD_H_
#define _GENERAL_GRAPH_OPTIMIZOR_SOLVER_GD_H_

#include "basic_type.h"
#include "solver.h"

namespace slam_solver {

/* Class Gradient Descent Solver Declaration. */
template <typename Scalar>
class SolverGd: public Solver<Scalar> {

public:
    struct SubOptions {
        Scalar kInitLearningRate = 1e-3;
        Scalar kMinLearningRate = 1e-12;
        Scalar kMaxLearningRate = 1e6;
        Scalar kLearningRateDecreaseFactor = 0.5;
        Scalar kLearningRateIncreaseFactor = 1.05;
    };

public:
    SolverGd(): Solver<Scalar>() {}
    virtual ~SolverGd() = default;

    // Initialize solver init value with first incremental function and so on.
    virtual void InitializeSolver() override;

    // Construct incremental function: use J and r directly (instead of H and b).
    virtual void ConstructIncrementalFunction(bool use_prior = false) override;

    // Compute gradient from J and r, then apply gradient descent step.
    virtual void SolveIncrementalFunction() override;

    // Check if one update step is valid.
    virtual bool IsUpdateValid(Scalar min_allowed_gain_rate = 0) override;

    // Reference for member variables.
    SubOptions &sub_options() { return sub_options_; }

    // Const reference for member variables.
    const SubOptions &sub_options() const { return sub_options_; }

private:
    // Options for GD solver.
    SubOptions sub_options_;

    // Parameters of GD solver.
    Scalar learning_rate_ = 0;
};

}  // namespace slam_solver

#endif  // end of _GENERAL_GRAPH_OPTIMIZOR_SOLVER_GD_H_
