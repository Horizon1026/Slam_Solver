#ifndef _SLAM_INVERSE_FILTER_H_
#define _SLAM_INVERSE_FILTER_H_

#include "basic_type.h"
#include "memory"

namespace SLAM_SOLVER {

/* Template Class InverseFilter Declaration. */
template <typename Scalar, typename InverseFilterType>
class InverseFilter {

public:
    InverseFilter() = default;
    virtual ~InverseFilter() = default;
    InverseFilter(const InverseFilter &filter) = delete;

    bool PropagateNominalState(const TVec<Scalar> &value = TMat<Scalar>::Zero(1, 1)) {
        return reinterpret_cast<InverseFilterType *>(this)->PropagateNominalStateImpl(value);
    }

    bool PropagateInformation() {
        return reinterpret_cast<InverseFilterType *>(this)->PropagateInformationImpl();
    }

    bool UpdateStateAndInformation(const TMat<Scalar> &value = TMat<Scalar>::Zero(1, 1)) {
        return reinterpret_cast<InverseFilterType *>(this)->UpdateStateAndInformationImpl(value);
    }

};

}

#endif // end of _SLAM_INVERSE_FILTER_H_
