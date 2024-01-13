#include "datatype_basic.h"
#include "log_report.h"
#include "visualizor.h"

#include "marginalizor.h"

using Scalar = float;
using namespace SLAM_UTILITY;
using namespace SLAM_SOLVER;
using namespace SLAM_VISUALIZOR;

void TestDiscardPriorInformation() {
    ReportInfo(YELLOW ">> Test discard prior information." RESET_COLOR);

    // Generate raw hessian and bias.
    TMat<Scalar> hessian = TMat6<Scalar>::Zero();
    TVec<Scalar> bias = TVec6<Scalar>::Zero();
    for (uint32_t i = 0; i < hessian.rows(); ++i) {
        hessian(i, i) = i + 1;
        bias(i) = i + 1;
    }
    ReportInfo("Prior hessian:\n" << hessian);
    ReportInfo("Prior bias:\n" << bias);

    // Do operation.
    Marginalizor<Scalar> marger;
    marger.DiscardPriorInformation(hessian, bias, 3, 4);
    ReportInfo("Prior hessian discarded:\n" << hessian);
    ReportInfo("Prior bias discarded:\n" << bias);
}

void TestMarginalizeOnlyHessian() {
    ReportInfo(YELLOW ">> Test marginalize only hessian and bias" RESET_COLOR);

    // Generate raw hessian and bias.
    TMat<Scalar> hessian = TMat6<Scalar>::Zero();
    TVec<Scalar> bias = TVec6<Scalar>::Zero();
    for (uint32_t i = 0; i < hessian.rows(); ++i) {
        hessian(i, i) = i + 1;
        bias(i) = i + 1;
    }
    ReportInfo("Prior hessian:\n" << hessian);
    ReportInfo("Prior bias:\n" << bias);

    // Do operation.
    Marginalizor<Scalar> marger;
    marger.options().kSortDirection = SortMargedVerticesDirection::kSortAtBack;
    marger.Marginalize(hessian, bias, 4, 1);
    ReportInfo("Prior hessian marged:\n" << hessian);
    ReportInfo("Prior bias marged:\n" << bias);
}

int main(int argc, char **argv) {
    LogFixPercision(3);
    ReportInfo(YELLOW ">> Test marginalizor operations." RESET_COLOR);

    TestDiscardPriorInformation();
    TestMarginalizeOnlyHessian();

    return 0;
}
