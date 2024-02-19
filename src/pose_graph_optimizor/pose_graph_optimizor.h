#ifndef _GENERAL_POSE_GRAPH_OPTIMIZOR_H_
#define _GENERAL_POSE_GRAPH_OPTIMIZOR_H_

#include "datatype_basic.h"

namespace SLAM_SOLVER {

/* Class Pose Graph Solver Declaration. */
template <typename Scalar>
class PoseGraphOptimizor {

public:
    explicit PoseGraphOptimizor() = default;
    virtual ~PoseGraphOptimizor() = default;

    // Reference for member variables.
    std::vector<TVec3<Scalar>> &all_p_wi() { return all_p_wi_; };
    std::vector<TQuat<Scalar>> &all_q_wi() { return all_q_wi_; };

    // Const Reference for member variables.
    const std::vector<TVec3<Scalar>> &all_p_wi() const { return all_p_wi_; };
    const std::vector<TQuat<Scalar>> &all_q_wi() const { return all_q_wi_; };

private:
    std::vector<TVec3<Scalar>> all_p_wi_;
    std::vector<TQuat<Scalar>> all_q_wi_;

};

}

#endif // end of _GENERAL_POSE_GRAPH_OPTIMIZOR_H_
