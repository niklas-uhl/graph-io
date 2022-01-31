#pragma once
#include <cassert>
#include <vector>
#include <utility>
#include <unordered_map>
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
    Indexer build_indexer() {
        return Indexer(*this);
    }
};

inline bool operator==(const LocalGraphView::NodeInfo& lhs, const LocalGraphView::NodeInfo& rhs) {
    return lhs.global_id == rhs.global_id && lhs.degree == rhs.degree;
}
}  // namespace graphio
