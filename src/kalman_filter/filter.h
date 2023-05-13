#ifndef _SLAM_FILTER_
#define _SLAM_FILTER_

#include "memory"

namespace SLAM_SOLVER {

/* Template Class Filter Declaration. */
template <typename Scalar, typename FilterType>
class Filter {

public:
    Filter() = default;
    virtual ~Filter() = default;

    bool PropagateState(const TVec<Scalar> &parameters = TVec<Scalar, 1>()) {
        return reinterpret_cast<FilterType *>(this)->Propagate(parameters);
    }

    bool UpdateState(const TMat<Scalar> &observation = TVec<Scalar, 1>()) {
        return reinterpret_cast<FilterType *>(this)->Update(observation);
    }

};

}

#endif
