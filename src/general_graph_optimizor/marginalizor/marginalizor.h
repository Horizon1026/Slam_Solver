#ifndef _GENERAL_GRAPH_OPTIMIZOR_MARGINALIZOR_H_
#define _GENERAL_GRAPH_OPTIMIZOR_MARGINALIZOR_H_

#include "datatype_basic.h"
#include "graph.h"
#include "log_api.h"

namespace SLAM_SOLVER {

template <typename Scalar>
struct MargOptions {

};

/* Class Marginalizor Declaration. */
template <typename Scalar>
class Marginalizor {

public:
    Marginalizor() = default;
    virtual ~Marginalizor() = default;

    // Marginalize graph optimization problem.
    bool Marginalize(bool use_prior = true);

    // Sort vertices to be marged to the front or back of vertices vector.
    // Keep the other vertices the same order.
    bool SortVerticesToBeMargedToTheFront(const std::vector<Vertex<Scalar> *> &vertices);
    bool SortVerticesToBeMargedToTheBack(const std::vector<Vertex<Scalar> *> &vertices);

    // Construct information.
    virtual void ConstructInformation(bool use_prior = true) = 0;

    // Create prior information.
    virtual void CreatePriorInformation() = 0;

private:
    // General options for marginalizor.
    MargOptions<Scalar> options_;

    // The graph optimization problem to be marged.
    Graph<Scalar> *problem_ = nullptr;

};

/* Class Marginalizor Definition. */
template <typename Scalar>
bool Marginalizor<Scalar>::Marginalize(bool use_prior) {
    if (problem_ == nullptr) {
        return false;
    }

    // Sort all vertices, determine their location in incremental function.
    problem_->SortVertices(false);

    return true;
}

}

#endif // end of _GENERAL_GRAPH_OPTIMIZOR_MARGINALIZOR_H_
