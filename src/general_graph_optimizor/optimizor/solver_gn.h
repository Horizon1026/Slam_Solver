#ifndef _GENERAL_GRAPH_OPTIMIZOR_SOLVER_GN_H_
#define _GENERAL_GRAPH_OPTIMIZOR_SOLVER_GN_H_

#include "basic_type.h"
#include "solver.h"

namespace slam_solver {

/* Class Gauss-Newton Solver Declaration. */
template <typename Scalar>
class SolverGn: public Solver<Scalar> {

public:
    struct SubOptions {
        Scalar kStepScaleDecreaseFactor = 0.5;
        Scalar kMinStepScale = 1e-10;
    };

public:
    SolverGn(): Solver<Scalar>() {}
    virtual ~SolverGn() = default;

    // Initialize solver init value with first incremental function and so on.
    virtual void InitializeSolver() override;

    // Solve Hx=b, or Jx=-r, in order to get delta parameters.
    virtual void SolveIncrementalFunction() override;

    // Check if one update step is valid.
    virtual bool IsUpdateValid(Scalar min_allowed_gain_rate = 0) override;

    // Reference for member variables.
    SubOptions &sub_options() { return sub_options_; }

    // Const reference for member variables.
    const SubOptions &sub_options() const { return sub_options_; }

private:
    // Options for GN solver.
    SubOptions sub_options_;

    // Parameters of GN solver.
    Scalar step_scale_ = 0;
    TVec<Scalar> dx_gn_;

    // Parameters of schur complement.
    TMat<Scalar> reverse_hessian_;
    TVec<Scalar> reverse_bias_;
    TVec<Scalar> reverse_dx_;
    TVec<Scalar> marg_bias_;
    TVec<Scalar> marg_dx_;
};

}  // namespace slam_solver

#endif  // end of _GENERAL_GRAPH_OPTIMIZOR_SOLVER_GN_H_
