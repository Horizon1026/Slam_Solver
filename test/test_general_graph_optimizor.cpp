#include "log_api.h"
#include "datatype_basic.h"
#include "vertex.h"

using Scalar = float;
using namespace SLAM_SOLVER;

int main(int argc, char **argv) {
    LogInfo(YELLOW ">> Test general graph optimizor." RESET_COLOR);

    Vertex<Scalar> vertex(3, 3);
    LogInfo("vertex type is " << vertex.GetType());

    return 0;
}
