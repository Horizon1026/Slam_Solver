#include "log_api.h"
#include "datatype_basic.h"
#include "vertex.h"
#include "edge.h"

using Scalar = float;
using namespace SLAM_SOLVER;

int main(int argc, char **argv) {
    LogInfo(YELLOW ">> Test general graph optimizor." RESET_COLOR);

    Vertex<Scalar> vertex(3, 3);
    LogInfo("vertex type is " << vertex.GetType());

    Edge<Scalar> edge(3, 3);
    LogInfo("edge type is " << edge.GetType());
    edge.SelfCheck();

    return 0;
}
