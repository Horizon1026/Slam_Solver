#ifndef _GENERAL_GRAPH_OPTIMIZOR_SOLVER_DOGLEG_H_
#define _GENERAL_GRAPH_OPTIMIZOR_SOLVER_DOGLEG_H_

#include "basic_type.h"
#include "solver.h"

namespace slam_solver {

template <typename Scalar>
struct SolverDoglegOptions {
    Scalar kInitRadius = 1e4;
    Scalar kMaxRadius = 1e8;
    Scalar kMinRadius = 1e-8;
};

/* Class Dogleg Solver Declaration. */
template <typename Scalar>
class SolverDogleg: public Solver<Scalar> {

public:
    SolverDogleg(): Solver<Scalar>() {}
    virtual ~SolverDogleg() = default;

    // Initialize solver init value with first incremental function and so on.
    virtual void InitializeSolver() override;

    // Solve Hx=b, or Jx=-r, in order to get delta parameters.
    virtual void SolveIncrementalFunction() override;

    // Check if one update step is valid.
    virtual bool IsUpdateValid(Scalar min_allowed_gain_rate = 0) override;

    // Reference for member variables.
    SolverDoglegOptions<Scalar> &dogleg_options() { return dogleg_options_; }

    // Const reference for member variables.
    const SolverDoglegOptions<Scalar> &dogleg_options() const { return dogleg_options_; }

private:
    // Options for Dogleg solver.
    SolverDoglegOptions<Scalar> dogleg_options_;

    // Parameters of Dogleg solver.
    Scalar radius_ = 0;
    Scalar alpha_ = 0;

    // Parameters of schur complement.
    TVec<Scalar> dx_sd_;
    TVec<Scalar> dx_gn_;
    TVec<Scalar> diff_dx_sd_gn_;
    TMat<Scalar> reverse_hessian_;
    TVec<Scalar> reverse_bias_;
    TVec<Scalar> reverse_dx_;
    TVec<Scalar> marg_bias_;
    TVec<Scalar> marg_dx_;
};

}  // namespace slam_solver

#endif  // end of _GENERAL_GRAPH_OPTIMIZOR_SOLVER_DOGLEG_H_
