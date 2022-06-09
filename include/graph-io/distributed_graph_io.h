#pragma once

#include <mpi.h>
#include <unordered_set>
#include "graph-io/definitions.h"
#include "graph-io/graph_definitions.h"
#include "graph-io/local_graph_view.h"
#ifdef GRAPH_IO_USE_KAGEN
#include "graph-io/gen_parameters.h"
#endif

namespace graphio {
namespace internal {
using node_set = std::unordered_set<NodeId>;

struct GraphInfo {
    NodeId total_node_count;
    NodeId local_from;
    NodeId local_to;

    static GraphInfo even_distribution(NodeId total_node_count, PEID rank, PEID size) {
        GraphInfo graph_info;
        NodeId remaining_nodes = total_node_count % size;
        NodeId local_node_count =
            (total_node_count / size) + static_cast<NodeId>(static_cast<size_t>(rank) < remaining_nodes);
        NodeId local_from = (rank * local_node_count) +
                            static_cast<NodeId>(static_cast<size_t>(rank) >= remaining_nodes ? remaining_nodes : 0);
        NodeId local_to = local_from + local_node_count;
        graph_info.total_node_count = total_node_count;
        graph_info.local_from = local_from;
        graph_info.local_to = local_to;
        return graph_info;
    }

    NodeId local_node_count() const {
        return local_to - local_from;
    }
};

void read_metis_distributed(const std::string& input,
                            const GraphInfo& graph_info,
                            std::vector<Edge<>>& edge_list,
                            node_set& ghosts,
                            PEID rank,
                            PEID size);

void read_metis_distributed(const std::string& input,
                            const GraphInfo& graph_info,
                            std::vector<EdgeId>& first_out,
                            std::vector<NodeId>& head,
                            PEID rank,
                            PEID size);

void gather_PE_ranges(NodeId local_from,
                      NodeId local_to,
                      std::vector<std::pair<NodeId, NodeId>>& ranges,
                      const MPI_Comm& comm);

PEID get_PE_from_node_ranges(NodeId node, const std::vector<std::pair<NodeId, NodeId>>& ranges);

LocalGraphView read_local_metis_graph(const std::string& input, const GraphInfo& graph_info, PEID rank, PEID size);

LocalGraphView read_local_partitioned_edgelist(const std::string& input,
                                               PEID rank,
                                               PEID size);

std::pair<NodeId, NodeId> get_node_range(const std::string& input, PEID rank, PEID size);

LocalGraphView read_local_binary_graph(const std::string& input, const GraphInfo& graph_info, PEID rank, PEID size);

void read_graph_info_from_binary(const std::string& input, NodeId& node_count, EdgeId& edge_count);
}  // namespace internal

#ifdef GRAPH_IO_USE_KAGEN
LocalGraphView gen_local_graph(const GeneratorParameters& conf, PEID rank, PEID size);
#endif

LocalGraphView read_local_graph(const std::string& input, InputFormat format, PEID rank, PEID size);

void write_graph_view(const LocalGraphView& G, const std::string& output, PEID rank, PEID size);

LocalGraphView read_graph_view(const std::string& input, PEID rank, PEID size);

std::string dump_to_tmp(const LocalGraphView& G, PEID rank, PEID size);  // namespace cetric
}  // namespace graphio
