#ifndef _GENERAL_GRAPH_OPTIMIZOR_MARGINALIZOR_H_
#define _GENERAL_GRAPH_OPTIMIZOR_MARGINALIZOR_H_

#include "basic_type.h"
#include "slam_basic_math.h"
#include "slam_log_reporter.h"

#include "graph.h"
#include "vertex.h"

#include "vector"

namespace slam_solver {

enum SortMargedVerticesDirection : uint8_t {
    kSortAtFront = 0,
    kSortAtBack = 1,
};

template <typename Scalar>
struct MargOptions {
    SortMargedVerticesDirection kSortDirection = SortMargedVerticesDirection::kSortAtFront;
};

/* Class Marginalizor Declaration. */
template <typename Scalar>
class Marginalizor {

public:
    Marginalizor() = default;
    virtual ~Marginalizor() = default;

    // Marginalize graph optimization problem.
    bool Marginalize(std::vector<Vertex<Scalar> *> &vertices, bool use_prior = true);
    bool Marginalize(TMat<Scalar> &hessian, TVec<Scalar> &bias, uint32_t row_index, uint32_t dimension);


    // Decompose hessian and bias to be jacobian and residual.
    void DecomposeHessianAndBias(TMat<Scalar> &hessian, TVec<Scalar> &bias, TMat<Scalar> &jacobian, TVec<Scalar> &residual, TMat<Scalar> &jacobian_t_inv);

    // Discard specified cols and rows of hessian and bias.
    bool DiscardPriorInformation(TMat<Scalar> &hessian, TVec<Scalar> &bias, uint32_t row_index, uint32_t dimension);

    // Reference for member variables.
    MargOptions<Scalar> &options() { return options_; }
    Graph<Scalar> *&problem() { return problem_; }
    Scalar &cost_of_problem() { return cost_of_problem_; }
    TMat<Scalar> &reverse_hessian() { return reverse_hessian_; }
    TVec<Scalar> &reverse_bias() { return reverse_bias_; }

    // Const reference for member variables.
    const MargOptions<Scalar> &options() const { return options_; }
    const Graph<Scalar> *problem() const { return problem_; }
    const Scalar &cost_of_problem() const { return cost_of_problem_; }
    const TMat<Scalar> &reverse_hessian() const { return reverse_hessian_; }
    const TVec<Scalar> &reverse_bias() const { return reverse_bias_; }

private:
    // Construct information.
    void ConstructInformation(bool use_prior = true);

    // Marginalize sparse vertices in information.
    void MarginalizeSparseVertices();

    // Create prior information, decompose hessian and bias, and store them in graph problem.
    void CreatePriorInformation(int32_t reverse_size, int32_t marge_size);

    // Create prior information, but only hessian and bias.
    void CreatePriorInformationOnlyHessianAndBias(const TMat<Scalar> &hessian, const TVec<Scalar> &bias, int32_t reverse_size, int32_t marge_size);

    // Compute prior information with schur complement.
    void ComputePriorBySchurComplement(const TMat<Scalar> &Hrr, const TMat<Scalar> &Hrm, const TMat<Scalar> &Hmr, const TMat<Scalar> &Hmm,
                                       const TVec<Scalar> &br, const TVec<Scalar> &bm);

    // Do schur complement.
    void SchurComplement(const TMat<Scalar> &Hrr, const TMat<Scalar> &Hrm, const TMat<Scalar> &Hmr, const TMat<Scalar> &Hmm, const TVec<Scalar> &br,
                         const TVec<Scalar> &bm, TMat<Scalar> &hessian, TVec<Scalar> &bias) const;

    // Move the matrix block which needs to be margnalized to the bound of matrix.
    bool MoveMatrixBlocksNeedMarginalization(TMat<Scalar> &hessian, TVec<Scalar> &bias, uint32_t row_index, uint32_t dimension);

private:
    // General options for marginalizor.
    MargOptions<Scalar> options_;

    // The graph optimization problem to be marged.
    Graph<Scalar> *problem_ = nullptr;
    Scalar cost_of_problem_ = 0.0;

    // The size of vertices needing to be marged.
    int32_t size_of_vertices_need_marge_ = 0;

    // Parameters of schur complement.
    TMat<Scalar> reverse_hessian_;
    TVec<Scalar> reverse_bias_;
};

}  // namespace slam_solver

#endif  // end of _GENERAL_GRAPH_OPTIMIZOR_MARGINALIZOR_H_
