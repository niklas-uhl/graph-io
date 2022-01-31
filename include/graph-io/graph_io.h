#pragma once

#include "graph-io/graph.h"
#include "graph-io/definitions.h"
#ifdef GRAPH_IO_PARALLEL
#include "graph-io/distributed_graph_io.h"
#endif

namespace graphio {
    Graph read_graph(const std::string& input, InputFormat format);
}
