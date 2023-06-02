#ifndef _GENERAL_GRAPH_OPTIMIZOR_MARGINALIZOR_H_
#define _GENERAL_GRAPH_OPTIMIZOR_MARGINALIZOR_H_

#include "datatype_basic.h"
#include "math_kinematics.h"
#include "log_report.h"

#include "vertex.h"
#include "graph.h"

#include "vector"

namespace SLAM_SOLVER {

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
    bool Marginalize(std::vector<Vertex<Scalar> *> &vertices,
                     bool use_prior = true);

    // Sort vertices to be marged to the front or back of vertices vector.
    // Keep the other vertices the same order.
    virtual void SortVerticesToBeMarged(std::vector<Vertex<Scalar> *> &vertices);

    // Construct information.
    virtual void ConstructInformation(bool use_prior = true);

    // Marginalize sparse vertices in information.
    virtual void MarginalizeSparseVertices();

    // Create prior information, and store them in graph problem.
    virtual void CreatePriorInformation();

    // Reference for member varibles.
    MargOptions<Scalar> &options() { return options_; }
    Graph<Scalar> *&problem() { return problem_; }

    // Const reference for member varibles.
    const MargOptions<Scalar> &options() const { return options_; }
    const Graph<Scalar> *problem() const { return problem_; }

private:
    // Compute prior information with schur complement.
    void ComputePriorBySchurComplement(const TMat<Scalar> &Hrr,
                                       const TMat<Scalar> &Hrm,
                                       const TMat<Scalar> &Hmr,
                                       const TMat<Scalar> &Hmm,
                                       const TVec<Scalar> &br,
                                       const TVec<Scalar> &bm);

private:
    // General options for marginalizor.
    MargOptions<Scalar> options_;

    // The graph optimization problem to be marged.
    Graph<Scalar> *problem_ = nullptr;

    // The size of vertices needing to be marged.
    int32_t size_of_vertices_need_marge_ = 0;

    // Parameters of schur complement.
    TMat<Scalar> reverse_hessian_;
    TVec<Scalar> reverse_bias_;

};

}

#endif // end of _GENERAL_GRAPH_OPTIMIZOR_MARGINALIZOR_H_
