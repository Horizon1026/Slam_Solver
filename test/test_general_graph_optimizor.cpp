#include "log_api.h"
#include "datatype_basic.h"

#include "vertex.h"
#include "edge.h"
#include "graph.h"
#include "solver.h"
#include "solver_lm.h"

using Scalar = float;
using namespace SLAM_SOLVER;

/* Class Vertex param a, b, c */
class VertexParam : public Vertex<Scalar> {

public:
    VertexParam() = delete;
    VertexParam(int32_t param_dim, int32_t delta_dim) : Vertex<Scalar>(param_dim, delta_dim) {}
    virtual ~VertexParam() = default;

    virtual std::string GetType() override { return std::string("Vertex a, b, c"); }
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
        auto vertex = this->GetVertex(0);
        a_ = vertex->param().x();
        b_ = vertex->param().y();
        c_ = vertex->param().z();

        // Compute residual.
        TVec<Scalar> res = Eigen::Matrix<Scalar, 1, 1>(a_ * x_ * x_ * x_ + b_ * x_ * x_ + c_ * x_ - y_);
        this->residual() = res;
    }

    virtual void ComputeJacobians() override {
        // Compute jacobian.
        auto &jacobian = this->GetJacobian(0);
        jacobian << x_ * x_ * x_,
                    x_ * x_,
                    x_;
    }

    virtual std::string GetType() override { return std::string("Edge r = y - (a * x^3 + b * x^2 + c * x)"); }

private:
    Scalar x_, y_, a_, b_, c_;

};

constexpr int32_t kMaxSampleNum = 100;

int main(int argc, char **argv) {
    LogInfo(YELLOW ">> Test general graph optimizor." RESET_COLOR);
    TVec3<Scalar> ground_truth_param = TVec3<Scalar>(2, 3, 4);
    Scalar a = ground_truth_param(0);
    Scalar b = ground_truth_param(1);
    Scalar c = ground_truth_param(2);
    LogInfo("Ground truth is a[" << a << "], b[" << b << "], c[" << c << "]");

    std::array<std::unique_ptr<VertexParam>, 1> vertices = {};
    vertices[0] = std::make_unique<VertexParam>(3, 3);
    vertices[0]->param() = TVec3<Scalar>(a + 0.5, b + 0.5, c + 0.5);

    std::array<std::unique_ptr<EdgePolynomial>, kMaxSampleNum> edges = {};
    for (int32_t i = 0; i < kMaxSampleNum; ++i) {
        edges[i] = std::make_unique<EdgePolynomial>(1, 1);
        edges[i]->SetVertex(vertices[0].get(), 0);

        Scalar x = i - kMaxSampleNum / 2;
        TVec2<Scalar> obv = TVec2<Scalar>(x, a * x * x * x + b * x * x + c * x);
        edges[i]->observation() = obv;
        edges[i]->SelfCheck();
    }

    SolverLm<Scalar> solver;
    for (auto &vertex : vertices) { solver.problem().AddVertex(vertex.get()); }
    for (auto &edge : edges) { solver.problem().AddEdge(edge.get()); }
    solver.Solve(false);

    LogInfo("Solve result is " << LogVec(vertices[0]->param()));

    return 0;
}
