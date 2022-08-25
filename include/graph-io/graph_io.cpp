#include "graph-io/graph_io.h"
#include <cstddef>
#include <numeric>
#include <stdexcept>
#include "graph-io/graph_definitions.h"
#include "graph-io/parsing.h"

namespace graphio {
Graph read_graph(const std::string& input, InputFormat format) {
    Graph G;
    if (format == InputFormat::metis) {
        NodeId edge_count = 0;
        internal::read_metis(
            input,
            [&](NodeId node_count, EdgeId edge_count) {
                G.first_out_.reserve(node_count + 1);
                G.head_.reserve(edge_count);
            },
            [&](NodeId node) { G.first_out_.emplace_back(edge_count); },
            [&](Edge<> edge) {
                G.head_.emplace_back(edge.head);
                edge_count++;
            });
        G.first_out_.emplace_back(edge_count);
    } else if (format == InputFormat::edge_list) {
        std::vector<Edge<>> edges;
        NodeId max_node_id = 0;
        internal::read_edge_list(input, [&](Edge<> edge) {
            max_node_id = std::max(std::max(edge.head, edge.tail), max_node_id);
            edges.push_back(edge);
        });
        G.first_out_.resize(max_node_id + 2);
        for (const auto& edge : edges) {
            G.first_out_[edge.tail]++;
        }
        std::exclusive_scan(G.first_out_.begin(), G.first_out_.end(), G.first_out_.begin(), EdgeId{0});
        G.head_.resize(edges.size());
        for (const auto& edge : edges) {
            auto& pos = G.first_out_[edge.tail];
            G.head_[pos] = edge.head;
            pos++;
        }
        edges.clear();
        edges.resize(0);
        for (size_t i = G.first_out_.size() - 1; i > 0; i--) {
            G.first_out_[i] = G.first_out_[i - 1];
        }
        G.first_out_[0] = 0;
    } else if (format == InputFormat::binary) {
        internal::read_binary<false>(input, G.first_out_, G.head_);
    } else {
        throw std::runtime_error("Unsupported format");
    }
    return G;
}
}  // namespace graphio
