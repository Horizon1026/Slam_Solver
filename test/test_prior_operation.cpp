#include "datatype_basic.h"
#include "log_report.h"
#include "visualizor.h"

#include "marginalizor.h"

using Scalar = float;
using namespace SLAM_UTILITY;
using namespace SLAM_SOLVER;
using namespace SLAM_VISUALIZOR;

int main(int argc, char **argv) {
    LogFixPercision(3);
    ReportInfo(YELLOW ">> Test marginalizor operations." RESET_COLOR);

    TMat<Scalar> hessian = TMat6<Scalar>::Zero();
    TVec<Scalar> bias = TVec6<Scalar>::Zero();
    for (uint32_t i = 0; i < hessian.rows(); ++i) {
        hessian(i, i) = i + 1;
        bias(i) = i + 1;
    }
    ReportInfo("Prior hessian:\n" << hessian);
    ReportInfo("Prior bias:\n" << bias);

    Marginalizor<Scalar> marger;
    marger.DiscardPriorInformation(hessian, bias, 3, 4);
    ReportInfo("Prior hessian discarded:\n" << hessian);
    ReportInfo("Prior bias discarded:\n" << bias);

    return 0;
}
