/**
 * @file test_state_filter_consistency.cpp
 * @brief Consistency test for all Kalman filter variants (Static + Dynamic).
 *
 * Given a non-triangular square-root state covariance matrix S_t,
 * an initial state, an observation, observation covariance, and
 * a null-space projection direction, each Kalman filter variant
 * performs one update step. The error-state Kalman filter (ESKF)
 * is used as the baseline; all other filter outputs are compared
 * against it.
 *
 * Dimensions:
 *   State size       = 4
 *   Observation size = 2
 *
 * Covariance is always reported in standard form P, where:
 *   - ESKF / KF:              P stored directly
 *   - Square Root KF:         P = S_t * S_t^T  (S_t is S^T)
 *   - Information Filter / EIF:  P = I^{-1}
 *   - Square Root IF:         P = (W^T * W)^{-1}
 *
 * Dependencies: only Eigen (provided via Slam_Utility/basic_type) and
 *               the filter headers — no other third-party library.
 */

#include <string>

#include "slam_log_reporter.h"
#include "error_information_filter.h"
#include "error_kalman_filter.h"
#include "information_filter.h"
#include "kalman_filter.h"
#include "square_root_information_filter.h"
#include "square_root_kalman_filter.h"

using Scalar = double;
using namespace slam_solver;

static constexpr int kStateSize = 4;
static constexpr int kObsSize   = 2;

// ---------------------------------------------------------------------------
//  Structure to hold one filter's result
// ---------------------------------------------------------------------------
struct FilterResult {
    std::string name;
    TVec<Scalar, kStateSize> dx;
    TMat<Scalar, kStateSize, kStateSize> P;
};

// ---------------------------------------------------------------------------
//  Shared test data (inputs + derived quantities)
// ---------------------------------------------------------------------------
struct TestData {
    TMat<Scalar, kStateSize, kStateSize>      S_t;
    TVec<Scalar, kStateSize>                  x_nominal;
    TVec<Scalar, kObsSize>                    residual;
    TMat<Scalar, kObsSize, kStateSize>        H;
    TMat<Scalar, kObsSize, kObsSize>          R;
    TMat<Scalar, kStateSize, Eigen::Dynamic>  null_space;

    TMat<Scalar, kStateSize, kStateSize>      P;
    TMat<Scalar, kStateSize, kStateSize>      I_mat;
    TMat<Scalar, kObsSize, kObsSize>          sqrt_R_t;
    TMat<Scalar, kObsSize, kObsSize>          inv_sqrt_R_t;
    TMat<Scalar, kStateSize, kStateSize>      Q;
    TMat<Scalar, kStateSize, kStateSize>      sqrt_Q_t;
    TMat<Scalar, kStateSize, kStateSize>      inv_sqrt_Q_t;
    TVec<Scalar, kObsSize>                    observation;
};

static TestData PrepareData(
    const TMat<Scalar, kStateSize, kStateSize>     &S_t,
    const TVec<Scalar, kStateSize>                  &x_nominal,
    const TVec<Scalar, kObsSize>                    &residual,
    const TMat<Scalar, kObsSize, kStateSize>        &H,
    const TMat<Scalar, kObsSize, kObsSize>          &R,
    const TMat<Scalar, kStateSize, Eigen::Dynamic>  &null_space) {

    TestData d;
    d.S_t       = S_t;
    d.x_nominal = x_nominal;
    d.residual  = residual;
    d.H         = H;
    d.R         = R;
    d.null_space = null_space;

    d.P     = S_t.transpose() * S_t;
    d.I_mat = d.P.inverse();

    // Decompose R
    Eigen::LLT<TMat<Scalar, kObsSize, kObsSize>> llt_R(R);
    TMat<Scalar, kObsSize, kObsSize> L_R = llt_R.matrixL();
    d.sqrt_R_t = L_R.transpose();
    {
        TMat<Scalar, kObsSize, kObsSize> inv_sqrt_R =
            L_R.triangularView<Eigen::Lower>().solve(
                TMat<Scalar, kObsSize, kObsSize>::Identity());
        d.inv_sqrt_R_t = inv_sqrt_R.transpose();
    }

    // Process noise
    constexpr Scalar kProcNoise = 1e-4;
    d.Q = TMat<Scalar, kStateSize, kStateSize>::Identity() * kProcNoise;

    Eigen::LLT<TMat<Scalar, kStateSize, kStateSize>> llt_Q(d.Q);
    TMat<Scalar, kStateSize, kStateSize> L_Q = llt_Q.matrixL();
    d.sqrt_Q_t = L_Q.transpose();
    {
        TMat<Scalar, kStateSize, kStateSize> inv_sqrt_Q =
            L_Q.triangularView<Eigen::Lower>().solve(
                TMat<Scalar, kStateSize, kStateSize>::Identity());
        d.inv_sqrt_Q_t = inv_sqrt_Q.transpose();
    }

    d.observation = H * x_nominal + residual;
    return d;
}

// ---------------------------------------------------------------------------
//  Print a comparison table of all filters against the baseline
// ---------------------------------------------------------------------------
static void PrintComparison(const std::string &group_name,
                            const FilterResult &baseline,
                            const FilterResult *others, int n_others) {

    LogFixPercision(7);

    ReportInfo(group_name + " — comparison against ESKF baseline");

    // ----  dx table  ----------------------------------------------------
    ReportText("--- State update dx ---\n");
    ReportText(std::left << std::setw(30) << "Filter"
              << "dx[0]           dx[1]           dx[2]           dx[3]          ||diff||\n");
    ReportText(std::string(105, '-') << "\n");

    auto print_vec = [](const auto &v) {
        Eigen::IOFormat vf(7, 0, ", ", ", ", "", "", "[", "]");
        ReportText(v.transpose().format(vf));
    };

    ReportText(std::left << std::setw(30) << baseline.name);
    print_vec(baseline.dx);
    ReportText("        ——\n");

    for (int i = 0; i < n_others; ++i) {
        Scalar d = (others[i].dx - baseline.dx).norm();
        ReportText(std::left << std::setw(30) << others[i].name);
        print_vec(others[i].dx);
        ReportText("   " << d << "\n");
    }

    // ----  P table  -----------------------------------------------------
    ReportText("\n--- Covariance P (max element diff / Frobenius norm diff) ---\n");
    ReportText(std::left << std::setw(30) << "Filter"
              << " max|dP_ij|    ||dP||_F\n");
    ReportText(std::string(65, '-') << "\n");
    for (int i = 0; i < n_others; ++i) {
        auto dP = others[i].P - baseline.P;
        ReportText(std::left << std::setw(30) << others[i].name
                  << dP.cwiseAbs().maxCoeff() << "    "
                  << dP.norm() << "\n");
    }
    ReportEndLine();
}

// ---------------------------------------------------------------------------
//  Templated test: true  -> Static variants
//                  false -> Dynamic variants
// ---------------------------------------------------------------------------
template <bool IsStatic>
static void RunOne(const TestData &d, const std::string &test_label) {

    using ESKF = std::conditional_t<IsStatic,
        ErrorKalmanFilterStatic<Scalar, kStateSize, kObsSize>,
        ErrorKalmanFilterDynamic<Scalar>>;
    using KF = std::conditional_t<IsStatic,
        KalmanFilterStatic<Scalar, kStateSize, kObsSize>,
        KalmanFilterDynamic<Scalar>>;
    using SRKF = std::conditional_t<IsStatic,
        SquareRootKalmanFilterStatic<Scalar, kStateSize, kObsSize>,
        SquareRootKalmanFilterDynamic<Scalar>>;
    using IF = std::conditional_t<IsStatic,
        InformationFilterStatic<Scalar, kStateSize, kObsSize>,
        InformationFilterDynamic<Scalar>>;
    using EIF = std::conditional_t<IsStatic,
        ErrorInformationFilterStatic<Scalar, kStateSize, kObsSize>,
        ErrorInformationFilterDynamic<Scalar>>;
    using SRIF = std::conditional_t<IsStatic,
        SquareRootInformationFilterStatic<Scalar, kStateSize, kObsSize>,
        SquareRootInformationFilterDynamic<Scalar>>;

    const std::string variant = IsStatic ? "Static" : "Dynamic";
    ReportInfo("===== " + test_label + "  [" + variant + "] =====");

    auto setIden = [](int n) { return TMat<Scalar>::Identity(n, n); };

    // ====================================================================
    //  1.  ESKF  (baseline)
    // ====================================================================
    ESKF eskf;
    eskf.options().kMethod = StateCovUpdateMethod::kFull;
    eskf.P() = d.P;
    eskf.F() = setIden(kStateSize);
    eskf.H() = d.H;
    eskf.Q() = d.Q;
    eskf.R() = d.R;
    eskf.null_space() = d.null_space;
    eskf.PropagateCovariance();
    eskf.UpdateStateAndCovariance(d.residual);

    FilterResult baseline;
    baseline.name = "ESKF";
    baseline.dx   = eskf.dx();
    baseline.P    = eskf.P();

    // ====================================================================
    //  2.  Standard KF  (full state)
    // ====================================================================
    KF kf;
    kf.options().kMethod = StateCovUpdateMethod::kFull;
    kf.P() = d.P;
    kf.x() = d.x_nominal;
    kf.F() = setIden(kStateSize);
    kf.H() = d.H;
    kf.Q() = d.Q;
    kf.R() = d.R;
    kf.null_space() = d.null_space;
    kf.PropagateCovariance();
    kf.UpdateStateAndCovariance(d.observation);

    FilterResult kf_res;
    kf_res.name = "KF";
    kf_res.dx   = kf.x() - kf.predict_x();
    kf_res.P    = kf.P();

    // ====================================================================
    //  3.  Square-Root KF  (error state)
    // ====================================================================
    SRKF srkf;
    srkf.options().kMethod = StateCovUpdateMethod::kFull;
    srkf.S_t() = d.S_t;
    srkf.F() = setIden(kStateSize);
    srkf.H()          = d.H;
    srkf.sqrt_Q_t()   = d.sqrt_Q_t;
    srkf.sqrt_R_t()   = d.sqrt_R_t;
    srkf.null_space() = d.null_space;
    srkf.PropagateCovariance();
    srkf.UpdateStateAndCovariance(d.residual);

    FilterResult srkf_res;
    srkf_res.name = "SRKF";
    srkf_res.dx   = srkf.dx();
    srkf_res.P    = srkf.S_t().transpose() * srkf.S_t();

    // ====================================================================
    //  4.  Information Filter  (full state)
    // ====================================================================
    IF inf;
    inf.x() = d.x_nominal;
    inf.I() = d.I_mat;
    inf.F() = setIden(kStateSize);
    inf.H()          = d.H;
    inf.inverse_Q()  = d.Q.inverse();
    inf.inverse_R()  = d.R.inverse();
    inf.null_space() = d.null_space;
    inf.PropagateInformation();
    inf.UpdateStateAndInformation(d.observation);

    FilterResult inf_res;
    inf_res.name = "IF";
    inf_res.dx   = inf.x() - inf.predict_x();
    inf_res.P    = inf.I().inverse();

    // ====================================================================
    //  5.  Error Information Filter
    // ====================================================================
    EIF eif;
    eif.I() = d.I_mat;
    eif.F() = setIden(kStateSize);
    eif.H()          = d.H;
    eif.inverse_Q()  = d.Q.inverse();
    eif.inverse_R()  = d.R.inverse();
    eif.null_space() = d.null_space;
    eif.PropagateInformation();
    eif.UpdateStateAndInformation(d.residual);

    FilterResult eif_res;
    eif_res.name = "EIF";
    eif_res.dx   = eif.dx();
    eif_res.P    = eif.I().inverse();

    // ====================================================================
    //  6.  Square-Root Information Filter
    // ====================================================================
    SRIF srif;
    {
        Eigen::LLT<TMat<Scalar, kStateSize, kStateSize>> llt(d.I_mat);
        // W must satisfy W^T * W = I (upper Cholesky), so that the QR-based
        // predict/update steps produce consistent results.  matrixU() gives
        // U where U^T * U = I_mat = P^{-1}.
        srif.W() = llt.matrixU();
    }
    srif.b() = TVec<Scalar>::Zero(kStateSize);
    srif.F() = setIden(kStateSize);
    srif.H()             = d.H;
    srif.inv_sqrt_Q_t()  = d.inv_sqrt_Q_t;
    srif.inv_sqrt_R_t()  = d.inv_sqrt_R_t;
    srif.null_space()    = d.null_space;
    srif.PropagateInformation();
    srif.UpdateStateAndInformation(d.residual);

    FilterResult srif_res;
    srif_res.name = "SRIF";
    srif_res.dx   = srif.dx();
    srif_res.P    = (srif.W().transpose() * srif.W()).inverse();

    // ----  Compare  -----------------------------------------------------
    FilterResult others[] = {kf_res, srkf_res, inf_res, eif_res, srif_res};
    PrintComparison(variant, baseline, others, 5);
}

// ===========================================================================
//  Main
// ===========================================================================
int main() {
    ReportInfo(YELLOW ">> Test state filter solver consistency." RESET_COLOR);

    // ----  Common test data  ------------------------------------------------
    // Non-triangular (full) square-root factor S_t;  P = S_t^T * S_t
    TMat<Scalar, kStateSize, kStateSize> S_t;
    S_t << 1.0,  0.2,  0.3,  0.1,
           0.4,  2.0,  0.1,  0.2,
           0.1,  0.3,  3.0,  0.3,
           0.2,  0.1,  0.2,  4.0;

    TVec<Scalar, kStateSize> x_nominal;
    x_nominal << 1.0, -0.5, 0.8, 0.3;

    TMat<Scalar, kObsSize, kStateSize> H;
    H << 1.0, 0.0, 1.0, 0.0,
         0.0, 1.0, 0.0, 1.0;

    TVec<Scalar, kObsSize> residual;
    residual << 1.5, -0.7;

    TMat<Scalar, kObsSize, kObsSize> R;
    R << 0.5, 0.1,
         0.1, 1.0;

    TMat<Scalar, kStateSize, 1> null_space;
    null_space << 1.0, 1.0, 0.0, 0.0;

    // ----  Test 1 : without null space  ------------------------------------
    {
        TMat<Scalar, kStateSize, Eigen::Dynamic> empty_ns(kStateSize, 0);
        TestData d = PrepareData(S_t, x_nominal, residual, H, R, empty_ns);
        RunOne<true>(d, "Test 1  —  NO null space");
        RunOne<false>(d, "Test 1  —  NO null space");
    }

    // ----  Test 2 : with null space  ---------------------------------------
    {
        TestData d = PrepareData(S_t, x_nominal, residual, H, R, null_space);
        RunOne<true>(d, "Test 2  —  WITH null space");
        RunOne<false>(d, "Test 2  —  WITH null space");
    }

    ReportInfo("All tests complete.");
    return 0;
}
