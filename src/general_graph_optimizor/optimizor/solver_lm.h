#ifndef _GENERAL_GRAPH_OPTIMIZOR_SOLVER_LM_H_
#define _GENERAL_GRAPH_OPTIMIZOR_SOLVER_LM_H_

#include "datatype_basic.h"
#include "solver.h"

namespace SLAM_SOLVER {

template <typename Scalar>
struct SolverLmOptions {
    Scalar kInitLambda = 1e-4;
};

/* Class LM Solver Declaration. */
template <typename Scalar>
class SolverLm : public Solver<Scalar> {

public:
    SolverLm() : Solver<Scalar>() {}
    virtual ~SolverLm() = default;

    // Initialize solver init value with first incremental function and so on.
    virtual void InitializeSolver() override;

    // Solve Hx=b, or Jx=-r, in order to get delta parameters.
    virtual void SolveIncrementalFunction() override;

    // Check if one update step is valid.
    virtual bool IsUpdateValid(Scalar min_allowed_gain_rate = 0) override;

    // Reference for member varibles.
    SolverLmOptions<Scalar> &lm_options() { return lm_options_; }

private:
    // Options for LM solver.
    SolverLmOptions<Scalar> lm_options_;

    // Parameters of LM solver.
    Scalar lambda_ = 0;
    Scalar v_ = 0;
    TVec<Scalar> diagnal_of_hessian_;
    TMat<Scalar> reverse_hessian_;
    TVec<Scalar> reverse_bias_;

};

/* Class LM Solver Definition. */
template <typename Scalar>
void SolverLm<Scalar>::InitializeSolver() {
    lambda_ = lm_options_.kInitLambda;
    // cost_at_linearized_point() = std::move(cost_at_latest_step());
    v_ = static_cast<Scalar>(2);
}

template <typename Scalar>
void SolverLm<Scalar>::SolveIncrementalFunction() {
    // Add diagnal of hessian.
    // diagnal_of_hessian_ = problem().hessian().diagonal();


    // Remove diagnal of hessian.
}

// Check if one update step is valid.
template <typename Scalar>
bool SolverLm<Scalar>::IsUpdateValid(Scalar min_allowed_gain_rate) {

    return true;
}

}

#endif
