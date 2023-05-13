#ifndef _SLAM_FILTER_
#define _SLAM_FILTER_

#include "datatype_basic.h"
#include "memory"

namespace SLAM_SOLVER {

enum StateCovUpdateMethod : uint8_t {
    kSimple = 0,
    kFull = 1,
};

/* Template Class Filter Declaration. */
template <typename Scalar, typename FilterType>
class Filter {

public:
    Filter() = default;
    virtual ~Filter() = default;
    Filter(const Filter &filter) = delete;

    bool PropagateState(const TVec<Scalar> &value = TVec<Scalar, 1>()) {
        return reinterpret_cast<FilterType *>(this)->Propagate(value);
    }

    bool UpdateState(const TMat<Scalar> &value = TVec<Scalar, 1>()) {
        return reinterpret_cast<FilterType *>(this)->Update(value);
    }

};

}

#endif
