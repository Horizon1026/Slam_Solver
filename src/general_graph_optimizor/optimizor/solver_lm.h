#ifndef _GENERAL_GRAPH_OPTIMIZOR_SOLVER_LM_H_
#define _GENERAL_GRAPH_OPTIMIZOR_SOLVER_LM_H_

#include "basic_type.h"
#include "solver.h"

namespace SLAM_SOLVER {

template <typename Scalar>
struct SolverLmOptions {
    Scalar kInitLambda = 1e-7;
    Scalar kMaxLambda = 1e32;
    Scalar kMinLambda = 1e-10;
};

/* Class LM Solver Declaration. */
template <typename Scalar>
class SolverLm : public Solver<Scalar> {

public:
    SolverLm()
        : Solver<Scalar>() {}
    virtual ~SolverLm() = default;

    // Initialize solver init value with first incremental function and so on.
    virtual void InitializeSolver() override;

    // Solve Hx=b, or Jx=-r, in order to get delta parameters.
    virtual void SolveIncrementalFunction() override;

    // Check if one update step is valid.
    virtual bool IsUpdateValid(Scalar min_allowed_gain_rate = 0) override;

    // Reference for member variables.
    SolverLmOptions<Scalar> &lm_options() { return lm_options_; }

    // Const reference for member variables.
    const SolverLmOptions<Scalar> &lm_options() const { return lm_options_; }

private:
    // Options for LM solver.
    SolverLmOptions<Scalar> lm_options_;

    // Parameters of LM solver.
    Scalar lambda_ = 0;
    Scalar v_ = 0;
    TVec<Scalar> diagnal_of_hessian_;

    // Parameters of schur complement.
    TMat<Scalar> reverse_hessian_;
    TVec<Scalar> reverse_bias_;
    TVec<Scalar> reverse_dx_;
    TVec<Scalar> marg_bias_;
    TVec<Scalar> marg_dx_;
};

}  // namespace SLAM_SOLVER

#endif  // end of _GENERAL_GRAPH_OPTIMIZOR_SOLVER_LM_H_
