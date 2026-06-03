#include "basic_type.h"
#include "slam_log_reporter.h"

#include "general_graph_optimizor.h"

using Scalar = double;
using namespace slam_solver;

/* Class Vertex param a, b, c */
class VertexParam: public Vertex<Scalar> {

public:
    VertexParam() = delete;
    VertexParam(int32_t param_dim, int32_t delta_dim): Vertex<Scalar>(param_dim, delta_dim) {}
    virtual ~VertexParam() = default;
};

/* Class Edge r = y - (a * x^3 + b * x^2 + c * x) */
class EdgePolynomial: public Edge<Scalar> {

public:
    EdgePolynomial() = delete;
    EdgePolynomial(int32_t residual_dim, int32_t vertex_num): Edge<Scalar>(residual_dim, vertex_num) {}
    virtual ~EdgePolynomial() = default;

    virtual void ComputeResidual() override {
        // Set observation.
        x_ = this->observation()(0, 0);
        y_ = this->observation()(1, 0);

        // Set param to be solved.
        a_ = this->GetVertex(0)->param()(0);
        b_ = this->GetVertex(1)->param()(0);
        c_ = this->GetVertex(2)->param()(0);

        // Compute residual.
        TVec<Scalar> res = Eigen::Matrix<Scalar, 1, 1>(a_ * x_ * x_ * x_ + b_ * x_ * x_ + c_ * x_ - y_);
        this->residual() = res;
    }

    // Provide analytical Jacobians for correctness (avoid numerical Jacobian issues).
    virtual void ComputeJacobians() override {
        // Compute jacobian: dr/da = x^3, dr/db = x^2, dr/dc = x
        this->GetJacobian(0) << x_ * x_ * x_;
        this->GetJacobian(1) << x_ * x_;
        this->GetJacobian(2) << x_;
    }

private:
    Scalar x_, y_, a_, b_, c_;
};

constexpr int32_t kMaxSampleNum = 100;

struct TestResult {
    std::string solver_name;
    TVec3<Scalar> result;
    Scalar final_cost;
};

// Template helper: run one solver test and record the result.
// Creates a fresh graph with polynomial fitting problem, runs the solver,
// and collects the final parameters and cost.
template <typename SolverType>
void RunSolverTest(const std::string &name, const TVec3<Scalar> &ground_truth, std::vector<TestResult> &results,
                   std::function<void(SolverType &)> configure = nullptr) {
    ReportInfo(GREEN ">> [" << name << "] Testing " << name << " solver." RESET_COLOR);

    // Create vertices and edges for this test.
    std::array<std::unique_ptr<VertexParam>, 3> vertices = {};
    std::array<std::unique_ptr<EdgePolynomial>, kMaxSampleNum> edges = {};

    // Setup vertices (initial params all zero).
    Graph<Scalar> problem;
    for (int32_t i = 0; i < 3; ++i) {
        vertices[i] = std::make_unique<VertexParam>(1, 1);
        vertices[i]->param() = TVec1<Scalar>(0);
        problem.AddVertex(vertices[i].get());
    }

    // Setup edges with synthetic observations from ground truth polynomial.
    const Scalar a_gt = ground_truth(0);
    const Scalar b_gt = ground_truth(1);
    const Scalar c_gt = ground_truth(2);
    for (int32_t i = 0; i < kMaxSampleNum; ++i) {
        edges[i] = std::make_unique<EdgePolynomial>(1, 3);
        edges[i]->SetVertex(vertices[0].get(), 0);
        edges[i]->SetVertex(vertices[1].get(), 1);
        edges[i]->SetVertex(vertices[2].get(), 2);

        const Scalar x = i - kMaxSampleNum / 2;
        TVec2<Scalar> obv = TVec2<Scalar>(x, a_gt * x * x * x + b_gt * x * x + c_gt * x);
        edges[i]->observation() = obv;
        edges[i]->SelfCheck();
        problem.AddEdge(edges[i].get());
    }

    // Create solver, apply optional configuration, then solve.
    SolverType solver;
    solver.problem() = &problem;
    if (configure) {
        configure(solver);
    }
    solver.Solve(true);

    // Extract and record result.
    TVec3<Scalar> result = TVec3<Scalar>(vertices[0]->param()(0), vertices[1]->param()(0), vertices[2]->param()(0));
    results.push_back({name, result, problem.ComputeResidualForAllEdges(true)});
    ReportInfo(name << " solve result is " << LogVec(result));
}

int main(int argc, char **argv) {
    ReportInfo(YELLOW ">> Test general graph optimizor on polynomial problem with multiple solvers." RESET_COLOR);
    const TVec3<Scalar> ground_truth_param(2, -3, -4);
    ReportInfo("Ground truth is " << LogVec(ground_truth_param));

    std::vector<TestResult> results;

    // Test each solver; GD gets extra iterations and a tuned learning rate.
    RunSolverTest<SolverLm<Scalar>>("Levenberg-Marquardt", ground_truth_param, results);
    RunSolverTest<SolverDogleg<Scalar>>("Dogleg", ground_truth_param, results);
    RunSolverTest<SolverGn<Scalar>>("Gauss-Newton", ground_truth_param, results);
    RunSolverTest<SolverGd<Scalar>>("Gradient Descent", ground_truth_param, results, [](SolverGd<Scalar> &solver) {
        solver.options().kMaxIteration = 100;
        solver.sub_options().kInitLearningRate = 0.5;
    });

    return 0;
}
