#include "datatype_basic.h"
#include "log_report.h"

#include "general_graph_optimizor.h"

using Scalar = float;
using namespace SLAM_SOLVER;

/* Class Vertex param a, b, c */
class VertexParam : public Vertex<Scalar> {

public:
    VertexParam() = delete;
    VertexParam(int32_t param_dim, int32_t delta_dim) : Vertex<Scalar>(param_dim, delta_dim) {}
    virtual ~VertexParam() = default;
};

/* Class Edge r = y - (a * x^3 + b * x^2 + c * x) */
class EdgePolynomial : public Edge<Scalar> {

public:
    EdgePolynomial() = delete;
    EdgePolynomial(int32_t residual_dim, int32_t vertex_num) : Edge<Scalar>(residual_dim, vertex_num) {}
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

    virtual void ComputeJacobians() override {
        // Compute jacobian.
        this->GetJacobian(0) << x_ * x_ * x_;
        this->GetJacobian(1) << x_ * x_;
        this->GetJacobian(2) << x_;
    }

private:
    Scalar x_, y_, a_, b_, c_;

};

constexpr int32_t kMaxSampleNum = 100;

int main(int argc, char **argv) {
    ReportInfo(YELLOW ">> Test general graph optimizor on polynomial problem." RESET_COLOR);
    TVec3<Scalar> ground_truth_param = TVec3<Scalar>(2, -3, -4);
    const Scalar a = ground_truth_param(0);
    const Scalar b = ground_truth_param(1);
    const Scalar c = ground_truth_param(2);
    ReportInfo("Ground truth is " << LogVec(ground_truth_param));

    std::array<std::unique_ptr<VertexParam>, 3> vertices = {};
    for (int32_t i = 0; i < 3; ++i) {
        vertices[i] = std::make_unique<VertexParam>(1, 1);
        vertices[i]->param() = TVec1<Scalar>(0);
    }

    std::array<std::unique_ptr<EdgePolynomial>, kMaxSampleNum> edges = {};
    for (int32_t i = 0; i < kMaxSampleNum; ++i) {
        edges[i] = std::make_unique<EdgePolynomial>(1, 3);
        edges[i]->SetVertex(vertices[0].get(), 0);
        edges[i]->SetVertex(vertices[1].get(), 1);
        edges[i]->SetVertex(vertices[2].get(), 2);

        const Scalar x = i - kMaxSampleNum / 2;
        TVec2<Scalar> obv = TVec2<Scalar>(x, a * x * x * x + b * x * x + c * x);
        edges[i]->observation() = obv;
        edges[i]->SelfCheck();
    }

    Graph<Scalar> problem;
    for (auto &vertex : vertices) { problem.AddVertex(vertex.get()); }
    for (auto &edge : edges) { problem.AddEdge(edge.get()); }
    SolverLm<Scalar> solver;
    solver.problem() = &problem;
    solver.Solve(true);

    TVec3<Scalar> result = TVec3<Scalar>(vertices[0]->param()(0), vertices[1]->param()(0), vertices[2]->param()(0));
    ReportInfo("Solve result is " << LogVec(result));

    return 0;
}
