#pragma once

#include <map>
#include <string>
#include "graph-io/graph_definitions.h"

namespace graphio {

struct GeneratorParameters {
    GeneratorParameters() = default;
    size_t seed = 28475421;
    std::string generator;
    NodeId n = 10;
    EdgeId m = 0;
    float r = 0.125;
    float r_coeff = 0.55;
    float p = 0.0;
    bool periodic = false;
    size_t k = 0;
    float gamma = 2.8;
    float d = 16;
    bool verify_graph = false;
    bool statistics = false;
};
}  // namespace graphio
