#ifndef _GENERAL_GRAPH_OPTIMIZOR_SOLVER_LM_H_
#define _GENERAL_GRAPH_OPTIMIZOR_SOLVER_LM_H_

#include "datatype_basic.h"
#include "solver.h"

namespace SLAM_SOLVER {

template <typename Scalar>
struct SolverLmOptions {
    Scalar kInitLambda = 1e-4;
    Scalar kMaxLambda = 1e32;
    Scalar kMinLambda = 1e-8;
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

    // Parameters of schur complement.
    TMat<Scalar> reverse_hessian_;
    TVec<Scalar> reverse_bias_;
    TVec<Scalar> reverse_dx_;
    TVec<Scalar> marg_bias_;
    TVec<Scalar> marg_dx_;

};

/* Class LM Solver Definition. */
template <typename Scalar>
void SolverLm<Scalar>::InitializeSolver() {
    // Use 'this->' to run function definded in basis class.
    lambda_ = lm_options_.kInitLambda;
    this->cost_at_linearized_point() = this->cost_at_latest_step();
    v_ = static_cast<Scalar>(2);
}

template <typename Scalar>
void SolverLm<Scalar>::SolveIncrementalFunction() {
    auto &hessian = this->problem().hessian();
    auto &bias = this->problem().bias();
    const int32_t hessian_size = hessian.rows();

    // Add diagnal of hessian.
    diagnal_of_hessian_ = hessian.diagonal();
    for (int32_t i = 0; i < hessian_size; ++i) {
        const Scalar temp = std::min(lm_options_.kMaxLambda, std::max(lm_options_.kMinLambda, hessian(i, i)));
        hessian(i, i) += lambda_ * temp;
    }

    // Solve incremental function.
    const int32_t reverse = this->problem().full_size_of_dense_vertices();
    const int32_t marg = this->problem().full_size_of_sparse_vertices();

    if (marg == 0) {
        // Directly solve the incremental function.
        this->SolveLinearlizedFunction(hessian, bias, this->dx());
    } else {
        this->dx().resize(hessian_size);

        // Firstly solve dense parameters.
        this->problem().MarginalizeSparseVerticesInHessianAndBias(reverse_hessian_, reverse_bias_);
        reverse_dx_.resize(reverse);
        this->SolveLinearlizedFunction(reverse_hessian_, reverse_bias_, reverse_dx_);
        this->dx().head(reverse) = reverse_dx_;

        // Secondly solve sparse parameters.
        marg_bias_ = bias.tail(marg) - hessian.block(reverse, 0, marg, reverse) * reverse_dx_;
        for (const auto &vertex : this->problem().sparse_vertices()) {
            const int32_t index = vertex->ColIndex();
            const int32_t dim = vertex->GetIncrementDimension();
            this->SolveLinearlizedFunction(hessian.block(index, index, dim, dim),
                                           marg_bias_.segment(index - reverse, dim),
                                           marg_dx_);
            this->dx().segment(index, dim) = marg_dx_;
        }
    }

    // Remove diagnal of hessian.
    hessian.diagonal() = diagnal_of_hessian_;
}

// Check if one update step is valid.
template <typename Scalar>
bool SolverLm<Scalar>::IsUpdateValid(Scalar min_allowed_gain_rate) {
    // Reference: The Levenberg-Marquardt method for nonlinear least squares curve-fitting problems.pdf
    TVec<Scalar> temp_vec = lambda_ * this->dx() + this->problem().bias();
    const Scalar scale = temp_vec.dot(this->dx()) + static_cast<Scalar>(1e-6);
    const Scalar rho = static_cast<Scalar>(0.5) * (
        this->cost_at_linearized_point() - this->cost_at_latest_step()
    ) / scale;

    // LogDebug("[Solver] lambda is " << lambda_ << ", rho is " << rho << ", cost is " << this->cost_at_latest_step() << "/" <<
    //     this->cost_at_linearized_point() << ", dx is " << this->dx().norm());

    bool result = true;
    if (rho > min_allowed_gain_rate && std::isfinite(this->cost_at_latest_step())) {
        lambda_ *= std::max(static_cast<Scalar>(0.33333),
                            static_cast<Scalar>((1.0 - std::pow(2.0 * rho - 1.0, 3))));
        v_ = static_cast<Scalar>(2);
        result = true;
    } else {
        lambda_ *= v_;
        v_ *= static_cast<Scalar>(2);
        result = false;
    }

    lambda_ = std::max(lm_options_.kMinLambda, std::min(lm_options_.kMaxLambda, lambda_));

    return result;
}

}

#endif
