#pragma once
#include <mpi.h>
#include <algorithm>
#include <array>
#include <cassert>
#include <iterator>
#include <iostream>
#include <sstream>
#include <unordered_map>
#include <utility>
#include <vector>
#include "graph-io/definitions.h"
#include "graph-io/graph_definitions.h"

namespace graphio {

struct LocalGraphView {
    struct NodeInfo {
        NodeInfo() = default;
        NodeInfo(NodeId global_id, Degree degree) : global_id(global_id), degree(degree){};
        NodeId global_id = 0;
        Degree degree = 0;
    };
    std::vector<NodeInfo> node_info;
    std::vector<NodeId> edge_heads;

    NodeId local_node_count() const {
        return node_info.size();
    }

    Degree degree(NodeId local_id) const {
        return node_info[local_id].degree;
    }

    class Indexer {
        friend LocalGraphView;

    public:
        template <typename NodeFunc>
        void for_each_neighbor(NodeId node_id, NodeFunc on_neighbor) const {
            NodeId index = get_index(node_id);
            assert(first_out[index + 1] - first_out[index] == G.node_info[index].degree);
            for (EdgeId edge_id = first_out[index]; edge_id < first_out[index + 1]; ++edge_id) {
                on_neighbor(G.edge_heads[edge_id]);
            }
        }

        std::pair<EdgeId, EdgeId> neighborhood_index_range(NodeId index) const {
            return std::make_pair(first_out[index], first_out[index + 1]);
        }
        bool has_node(NodeId node_id) const {
            return id_to_index.find(node_id) != id_to_index.end();
        }

        NodeId get_index(NodeId node_id) const {
            assert(id_to_index.find(node_id) != id_to_index.end());
            size_t i = 0;
            for (; i < G.node_info.size(); ++i) {
                if (G.node_info[i].global_id == node_id) {
                    break;
                }
            }
            assert(i < G.node_info.size());
            auto result = id_to_index.find(node_id)->second;
            assert(result == i);
            return result;
        }

    private:
        const LocalGraphView& G;
        std::vector<EdgeId> first_out;
        std::unordered_map<NodeId, NodeId> id_to_index;
        Indexer(const LocalGraphView& G) : G(G), first_out(G.node_info.size() + 1), id_to_index(G.node_info.size()) {
            // id_to_index.set_empty_key(-1);
            EdgeId prefix_sum = 0;
            for (size_t i = 0; i < G.node_info.size(); ++i) {
                first_out[i] = prefix_sum;
                prefix_sum += G.node_info[i].degree;
                id_to_index[G.node_info[i].global_id] = i;
                assert(id_to_index[G.node_info[i].global_id] == i);
            }
            first_out[G.node_info.size()] = prefix_sum;
        }
    };

    class NodeLocator {
        friend LocalGraphView;
    public:
        bool is_local(NodeId node) {
            return node >= local_range_.first && node <= local_range_.second;
        }

        PEID rank(NodeId node) {
            if (is_local(node)) {
                return rank_;
            }
            return rank_map[node];
        }

        std::pair<NodeId, NodeId> local_range() const {
            return local_range_;
        }

    private:
        NodeLocator(const LocalGraphView& G, MPI_Comm comm) : local_range_(), rank_map(), rank_() {
            bool is_sorted = std::is_sorted(G.node_info.begin(), G.node_info.end(),
                                            [](auto a, auto b) { return a.global_id < b.global_id; });
            if (!is_sorted) {
                throw "Node IDs must be globally sorted";
            }
            PEID rank, size;
            MPI_Comm_rank(comm, &rank);
            rank_ = rank;
            MPI_Comm_size(comm, &size);
            std::vector<std::pair<NodeId, NodeId>> ranges(size);
            local_range_ = {G.node_info.front().global_id, G.node_info.back().global_id};
            MPI_Allgather(&local_range_, 2, GRAPH_IO_MPI_NODE, ranges.data(), 2, GRAPH_IO_MPI_NODE, comm);
            for (NodeId node : G.edge_heads) {
                if (is_local(node)) {
                    continue;
                }
                if (rank_map.find(node) == rank_map.end()) {
                    auto it = std::lower_bound(
                        ranges.begin(), ranges.end(), node,
                        [](std::pair<NodeId, NodeId> range, NodeId node) { return range.second < node; });
                    rank_map[node] = std::distance(ranges.begin(), it);
                }
            }
        }
        std::pair<NodeId, NodeId> local_range_;
        std::unordered_map<NodeId, PEID> rank_map;
        PEID rank_;
    };
    Indexer build_indexer() {
        return Indexer(*this);
    }
    NodeLocator build_locator(MPI_Comm comm) {
        return NodeLocator(*this, comm);
    }
};

inline bool operator==(const LocalGraphView::NodeInfo& lhs, const LocalGraphView::NodeInfo& rhs) {
    return lhs.global_id == rhs.global_id && lhs.degree == rhs.degree;
}
}  // namespace graphio
