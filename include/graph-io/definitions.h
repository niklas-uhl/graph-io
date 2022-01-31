#pragma once
#include <map>
#include <string>

namespace graphio {
enum class InputFormat { metis, binary, partitioned_edgelist_dimacs, edge_list };

const std::map<std::string, InputFormat> input_types = {
    {"metis", InputFormat::metis},
    {"binary", InputFormat::binary},
    {"partitioned-dimacs", InputFormat::partitioned_edgelist_dimacs},
    {"edge-list", InputFormat::edge_list}};

#ifdef GRAPH_IO_PARALLEL
using PEID = int;
#endif
}  // namespace graphio
