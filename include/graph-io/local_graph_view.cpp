#include "graph-io/local_graph_view.h"


namespace graphio {

LocalGraphView apply_partition(LocalGraphView&& G, std::vector<size_t> const& partition, MPI_Comm comm) {
    PEID rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    assert(partition.size() == G.local_node_count());
    size_t number_of_partitions = *std::max_element(partition.begin(), partition.end()) + 1;
    MPI_Allreduce(MPI_IN_PLACE, &number_of_partitions, 1, MPI_UINT64_T, MPI_MAX, comm);
    assert(number_of_partitions <= static_cast<size_t>(size));
    std::vector<int> send_count_nodes(size);
    std::vector<int> send_count_edges(size);
    std::vector<int> send_displ_nodes(size);
    std::vector<int> send_displ_edges(size);
    for (size_t i = 0; i < G.node_info.size(); i++) {
        send_count_nodes[partition[i]]++;
        send_count_edges[partition[i]] += G.node_info[i].degree;
    }
    std::vector<int> recv_count_nodes(size);
    std::vector<int> recv_count_edges(size);
    std::vector<int> recv_displ_nodes(size);
    std::vector<int> recv_displ_edges(size);
    MPI_Alltoall(send_count_nodes.data(), 1, MPI_INT, recv_count_nodes.data(), 1, MPI_INT, comm);
    MPI_Alltoall(send_count_edges.data(), 1, MPI_INT, recv_count_edges.data(), 1, MPI_INT, comm);
    std::exclusive_scan(send_count_nodes.begin(), send_count_nodes.end(), send_displ_nodes.begin(), 0);
    std::exclusive_scan(send_count_edges.begin(), send_count_edges.end(), send_displ_edges.begin(), 0);
    std::exclusive_scan(recv_count_nodes.begin(), recv_count_nodes.end(), recv_displ_nodes.begin(), 0);
    std::exclusive_scan(recv_count_edges.begin(), recv_count_edges.end(), recv_displ_edges.begin(), 0);
    std::vector<LocalGraphView::NodeInfo> send_buf_nodes(G.node_info.size());
    std::vector<NodeId> send_buf_edges(G.edge_heads.size());
    auto idx = G.build_indexer();
    for (size_t i = 0; i < G.node_info.size(); i++) {
        auto insert_pos_node = send_displ_nodes[partition[i]];
        send_displ_nodes[partition[i]]++;
        send_buf_nodes[insert_pos_node] = G.node_info[i];

        auto insert_pos_edge = send_displ_edges[partition[i]];
        send_displ_edges[partition[i]] += G.node_info[i].degree;
        idx.for_each_neighbor(G.node_info[i].global_id, [&](NodeId neighbor) {
            send_buf_edges[insert_pos_edge] = neighbor;
            insert_pos_edge++;
        });
    }
    std::rotate(send_displ_nodes.rbegin(), send_displ_nodes.rbegin() + 1, send_displ_nodes.rend());
    send_displ_nodes[0] = 0; 
    std::rotate(send_displ_edges.rbegin(), send_displ_edges.rbegin() + 1, send_displ_edges.rend());
    send_displ_edges[0] = 0;
    MPI_Datatype node_info_type;
    MPI_Type_contiguous(2,GRAPH_IO_MPI_NODE, &node_info_type);
    MPI_Type_commit(&node_info_type);
    G.node_info.resize(recv_count_nodes.back() + recv_displ_nodes.back());
    MPI_Alltoallv(send_buf_nodes.data(), send_count_nodes.data(), send_displ_nodes.data(), node_info_type,
                  G.node_info.data(), recv_count_nodes.data(), recv_displ_nodes.data(), node_info_type, comm);
    MPI_Type_free(&node_info_type);

    // free the send_buf
    std::vector<LocalGraphView::NodeInfo>().swap(send_buf_nodes);

    G.edge_heads.resize(recv_count_edges.back() + recv_displ_edges.back());
    MPI_Alltoallv(send_buf_edges.data(), send_count_edges.data(), send_displ_edges.data(), GRAPH_IO_MPI_NODE,
                  G.edge_heads.data(), recv_count_edges.data(), recv_displ_edges.data(), GRAPH_IO_MPI_NODE, comm);
    std::vector<NodeId>().swap(send_buf_edges);
    return G;
}

void relabel_consecutively(LocalGraphView& G, MPI_Comm comm) {
    PEID rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    NodeId first_node = G.local_node_count();
    MPI_Exscan(MPI_IN_PLACE, &first_node, 1, GRAPH_IO_MPI_NODE, MPI_SUM, comm);
    NodeId total_node_count = first_node + G.local_node_count();
    MPI_Bcast(&total_node_count, 1, GRAPH_IO_MPI_NODE, size - 1, comm);
    if (rank == 0) {
        first_node = 0;
    }
    std::unordered_map<NodeId, NodeId> to_new_id;
    auto idx = G.build_indexer();
    for (size_t i = 0; i < G.node_info.size(); i++) {
        auto const& node_info = G.node_info[i];
        to_new_id[node_info.global_id] = first_node + i;
        idx.for_each_neighbor(node_info.global_id, [&](NodeId neighbor) {
            // check for ghost nodes
            if (!idx.has_node(neighbor)) {
                to_new_id[neighbor] = -1;
            }
        });
    }
    std::vector<NodeId> local_nodes(G.node_info.size());
    std::vector<NodeId> local_nodes_recv;
    std::transform(G.node_info.begin(), G.node_info.end(), local_nodes.begin(), [](auto node_info) {
        return node_info.global_id;
    });
    NodeId other_rank_first_node = first_node;
    for (int i = 1; i < size; i++) {
        MPI_Request req[2];
        MPI_Isend(local_nodes.data(), local_nodes.size(), GRAPH_IO_MPI_NODE, (rank + 1) % size, 0, comm, &req[0]);
        MPI_Status status;
        MPI_Probe((rank + size - 1) % size, 0, comm, &status);
        int recv_count;
        MPI_Get_count(&status, GRAPH_IO_MPI_NODE, &recv_count);
        local_nodes_recv.resize(recv_count);
        MPI_Irecv(local_nodes_recv.data(), local_nodes_recv.size(), GRAPH_IO_MPI_NODE, status.MPI_SOURCE, status.MPI_TAG, comm, &req[1]);
        MPI_Waitall(2, req, MPI_STATUSES_IGNORE);
        local_nodes.swap(local_nodes_recv);
        PEID other_rank = (rank + size - i) % size;
        other_rank_first_node = (other_rank_first_node + total_node_count - local_nodes.size()) % total_node_count;
        for (size_t j = 0; j < local_nodes.size(); j++) {
            NodeId const& old_id = local_nodes[j];
            NodeId new_id = other_rank_first_node + j;
            auto it = to_new_id.find(old_id);
            if (it != to_new_id.end()) {
                it->second = new_id;
            }
        }
    }
    std::for_each(G.node_info.begin(), G.node_info.end(), [&](LocalGraphView::NodeInfo& node_info) {
        node_info.global_id = to_new_id[node_info.global_id];
    });
    std::for_each(G.edge_heads.begin(), G.edge_heads.end(), [&](NodeId& node) {
        node = to_new_id[node];
    });
}

std::string as_dot(LocalGraphView const& G, MPI_Comm comm) {
    PEID size;
    MPI_Comm_size(comm, &size);

    std::stringstream out;
    auto idx = G.build_indexer();
    for (auto const& node_info : G.node_info) {
        idx.for_each_neighbor(node_info.global_id, [&](NodeId neighbor) {
            out << node_info.global_id << " -> " << neighbor << ";" << std::endl;
        });
    }
    std::string dot_string = out.str();
    int send_count = dot_string.size();
    std::vector<int> recv_count(size);
    MPI_Gather(&send_count, 1, MPI_INT, recv_count.data(), 1, MPI_INT, 0, comm);
    std::vector<int> displs(size);
    std::exclusive_scan(recv_count.begin(), recv_count.end(), displs.begin(), 0);
    std::string recv_buf;
    recv_buf.resize(recv_count.back() + displs.back());
    MPI_Gatherv(dot_string.data(), dot_string.size(), MPI_CHAR, recv_buf.data(), recv_count.data(), displs.data(), MPI_CHAR, 0, comm);
    std::stringstream full_string;
    full_string << "digraph G {" << std::endl;
    full_string << recv_buf;
    full_string << "}" << std::endl;
    return full_string.str();
}
}