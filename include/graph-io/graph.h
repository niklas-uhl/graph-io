#pragma once
#include <vector>
#include "graph-io/graph_definitions.h"

namespace graphio {
struct Graph {
    std::vector<EdgeId> first_out_;
    std::vector<NodeId> head_;
};
}  // namespace graphio
