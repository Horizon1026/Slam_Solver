#include "log_api.h"
#include "datatype_basic.h"
#include "vertex.h"
#include "edge.h"
#include "graph.h"
#include "solver.h"

using Scalar = float;
using namespace SLAM_SOLVER;

int main(int argc, char **argv) {
    LogInfo(YELLOW ">> Test general graph optimizor." RESET_COLOR);

    Vertex<Scalar> vertex(3, 3);
    LogInfo("vertex type is " << vertex.GetType());

    Edge<Scalar> edge(1, 1);
    LogInfo("edge type is " << edge.GetType());
    edge.SetVertex(&vertex, 0);
    edge.SelfCheck();

    Graph<Scalar> graph;
    graph.AddVertex(&vertex, true);
    graph.AddEdge(&edge);

    return 0;
}
