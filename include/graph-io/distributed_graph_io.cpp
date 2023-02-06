#include "graph-io/distributed_graph_io.h"
#include <mpi.h>
#include <unistd.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <unordered_set>
#include <utility>
#include "distributed_graph_io.h"
#include "graph-io/graph_definitions.h"
#include "graph-io/local_graph_view.h"
#include "graph-io/mpi_io_wrapper.h"
#include "graph-io/parsing.h"
#ifdef GRAPH_IO_USE_KAGEN
#include <kagen.h>
#include "graph-io/mpi_io_wrapper.h"
#endif
#include <fcntl.h>
#include <sys/mman.h>

namespace graphio {

namespace internal {
template <class EdgeList>
void fix_broken_edge_list(EdgeList& edge_list,
                          const std::vector<std::pair<NodeId, NodeId>>& ranges,
                          node_set& ghosts,
                          PEID rank,
                          PEID size) {
    NodeId local_from = ranges[rank].first;
    NodeId local_to = ranges[rank].second;

    std::unordered_map<PEID, std::vector<NodeId>> message_buffers;
    for (auto& edge : edge_list) {
        NodeId tail;
        NodeId head;
        if constexpr (!std::is_same<EdgeList, std::vector<Edge<>>>::value) {
            tail = edge.first;
            head = edge.second;
        } else {
            tail = edge.tail;
            head = edge.head;
        }
        if (tail >= local_from && tail < local_to) {
            if (head < local_from || head >= local_to) {
                message_buffers[get_PE_from_node_ranges(head, ranges)].emplace_back(tail);
                message_buffers[get_PE_from_node_ranges(head, ranges)].emplace_back(head);
            }
        }
    }
    std::vector<NodeId> send_buf;
    std::vector<NodeId> recv_buf;
    std::vector<int> send_counts(size);
    std::vector<int> recv_counts(size);
    std::vector<int> send_displs(size);
    std::vector<int> recv_displs(size);
    for (size_t i = 0; i < send_counts.size(); ++i) {
        send_counts[i] = message_buffers[i].size();
    }
    std::exclusive_scan(send_counts.begin(), send_counts.end(), send_displs.begin(), 0);
    size_t total_send_count = send_displs[size - 1] + send_counts[size - 1];
    MPI_Alltoall(send_counts.data(), 1, MPI_INT, recv_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);
    std::exclusive_scan(recv_counts.begin(), recv_counts.end(), recv_displs.begin(), 0);
    size_t total_recv_count = recv_displs[size - 1] + recv_counts[size - 1];
    send_buf.reserve(total_send_count);
    for (size_t i = 0; i < send_counts.size(); ++i) {
        for (auto elem : message_buffers[i]) {
            send_buf.push_back(elem);
        }
        message_buffers[i].clear();
        message_buffers[i].resize(0);
    }
    recv_buf.resize(total_recv_count);
    MPI_Alltoallv(send_buf.data(), send_counts.data(), send_displs.data(), MPI_UINT64_T, recv_buf.data(),
                  recv_counts.data(), recv_displs.data(), MPI_UINT64_T, MPI_COMM_WORLD);
    send_buf.clear();
    send_buf.resize(0);
    for (size_t i = 0; i < recv_buf.size(); i += 2) {
        edge_list.emplace_back(recv_buf[i + 1], recv_buf[i]);
        ghosts.insert(recv_buf[i]);
    }
    recv_buf.clear();
    recv_buf.resize(0);

    if constexpr (!std::is_same<EdgeList, std::vector<Edge<>>>::value) {
        std::sort(edge_list.begin(), edge_list.end());
    } else {
        std::sort(edge_list.begin(), edge_list.end(), [&](const Edge<>& e1, const Edge<>& e2) {
            return std::tie(e1.tail, e1.head) < std::tie(e2.tail, e2.head);
        });
    }

    // kagen sometimes produces duplicate edges
    auto it = std::unique(edge_list.begin(), edge_list.end());
    edge_list.erase(it, edge_list.end());
}

GraphInfo get_even_edge_distribution(const std::string& input, InputFormat format, PEID rank, PEID size) {
    NodeId total_node_count;
    EdgeId total_edge_count;
    GraphInfo info;
    if (format == InputFormat::metis) {
        if (rank == 0) {
            internal::read_metis_header(input, total_node_count, total_edge_count);
            total_edge_count *= 2;
            EdgeId edges_per_rank = (total_edge_count + size - 1) / size;
            std::vector<NodeId> first_node(size + 1);
            size_t running_sum = 0;
            PEID current_pe = -1;  // at the moment we are assigning tasks to this PE
            internal::read_metis(
                input, [](auto, auto) {},
                [&](NodeId node) {
                    PEID target_pe = running_sum / edges_per_rank;
                    while (current_pe < target_pe) {
                        current_pe++;
                        first_node[current_pe] = node;
                    }
                },
                [&](Edge<> edge) { running_sum++; });
            // some PEs might not get a vertices, fix them
            while (current_pe < size) {
                current_pe++;
                first_node[current_pe] = total_node_count;
            }
            MPI_Scatter(first_node.data() + 1, 1, GRAPH_IO_MPI_NODE, &info.local_to, 1, GRAPH_IO_MPI_NODE, 0,
                        MPI_COMM_WORLD);
        } else {
            MPI_Scatter(nullptr, 1, GRAPH_IO_MPI_NODE, &info.local_to, 1, GRAPH_IO_MPI_NODE, 0, MPI_COMM_WORLD);
        }
        PEID dest = rank < size - 1 ? rank + 1 : MPI_PROC_NULL;
        PEID source = rank == 0 ? MPI_PROC_NULL : rank - 1;
        MPI_Sendrecv(&info.local_to, 1, GRAPH_IO_MPI_NODE, dest, 0, &info.local_from, 1, GRAPH_IO_MPI_NODE, source, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if (rank == 0) {
            info.local_from = 0;
        }
    } else if (format == InputFormat::binary) {
        internal::read_graph_info_from_binary(input, total_node_count, total_edge_count);
        NodeId nodes_per_rank = (total_node_count + size - 1) / size;
        EdgeId edges_per_rank = (total_edge_count + size - 1) / size;
        auto input_path = std::filesystem::path(input);
        auto basename = input_path.stem();
        auto path = input_path.parent_path();
        auto first_out_path = path / (basename.string() + ".first_out");
        int fd = open(first_out_path.c_str(), O_RDONLY);
        if (fd < 0) {
            throw std::runtime_error("Failed to open " + first_out_path.string());
        };
        void* ptr = mmap(NULL, (total_node_count + 1) * sizeof(EdgeId), PROT_READ, MAP_PRIVATE, fd, 0);
        if (ptr == MAP_FAILED) {
            close(fd);
            throw std::runtime_error("Failed to map file " + first_out_path.string());
        }
        EdgeId* running_sum = static_cast<EdgeId*>(ptr) + rank * nodes_per_rank;
        EdgeId* running_sum_end = std::min(running_sum + nodes_per_rank, static_cast<EdgeId*>(ptr) + total_node_count);
        running_sum = std::min(running_sum, running_sum_end);
        std::vector<std::pair<PEID, NodeId>> first_node;
        PEID current_pe = rank == 0 ? -1 : *(running_sum - 1) / edges_per_rank;
        for (; running_sum < running_sum_end; running_sum++) {
            NodeId node = running_sum - static_cast<EdgeId*>(ptr);
            PEID target_pe = *running_sum / edges_per_rank;
            while (current_pe < target_pe) {
                current_pe++;
                if (current_pe < size) {
                    first_node.emplace_back(current_pe, node);
                }
            }
        }
        munmap(ptr, (total_node_count + 1) * sizeof(EdgeId));
        close(fd);
        if (rank == size - 1) {
            while (current_pe < size - 1) {
                current_pe++;
                if (current_pe < size) {
                    first_node.emplace_back(current_pe, total_node_count);
                }
            }
        }
        std::vector<MPI_Request> req(first_node.size());
        for (size_t i = 0; i < first_node.size(); i++) {
            MPI_Issend(&first_node[i].second, 1, GRAPH_IO_MPI_NODE, first_node[i].first, 0, MPI_COMM_WORLD,
                       req.data() + i);
        }
        // if (rank != 0) {
        MPI_Recv(&info.local_from, 1, GRAPH_IO_MPI_NODE, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
        // } else {
        // info.local_from = 0;
        // }
        MPI_Waitall(req.size(), req.data(), MPI_STATUSES_IGNORE);
        MPI_Barrier(MPI_COMM_WORLD);
        PEID dest = rank == 0 ? MPI_PROC_NULL : rank - 1;
        PEID source = rank < size - 1 ? rank + 1 : MPI_PROC_NULL;
        MPI_Sendrecv(&info.local_from, 1, GRAPH_IO_MPI_NODE, dest, 0, &info.local_to, 1, GRAPH_IO_MPI_NODE, source, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if (rank == size - 1) {
            info.local_to = total_node_count;
        }
    } else {
        throw std::runtime_error("Unsupported format");
    }
    return info;
}

void read_metis_distributed(const std::string& input,
                            const GraphInfo& graph_info,
                            std::vector<Edge<>>& edge_list,
                            node_set& ghosts,
                            PEID rank,
                            PEID size) {
    (void)rank;
    (void)size;

    auto on_head = [](NodeId, EdgeId) {
    };

    bool skip = true;
    auto on_node = [&](NodeId node) {
        if (node >= graph_info.local_from) {
            skip = false;
        }
        if (node >= graph_info.local_to) {
            skip = true;
        }
    };

    auto on_edge = [&](Edge<> edge) {
        if (!skip) {
            if (edge.head < graph_info.local_from || edge.head >= graph_info.local_to) {
                ghosts.insert(edge.head);
            }
            edge_list.emplace_back(edge);
        }
    };

    internal::read_metis(input, on_head, on_node, on_edge);
}

void read_metis_distributed(const std::string& input,
                            const GraphInfo& graph_info,
                            std::vector<EdgeId>& first_out,
                            std::vector<NodeId>& head,
                            PEID rank,
                            PEID size) {
    (void)rank;
    (void)size;

    auto on_head = [](NodeId, EdgeId) {
    };

    bool skip = true;
    auto on_node = [&](NodeId node) {
        if (node >= graph_info.local_from && node < graph_info.local_to) {
            first_out.emplace_back(head.size());
            skip = false;
        } else {
            skip = true;
        }
    };

    auto on_edge = [&](Edge<> edge) {
        if (!skip) {
            head.emplace_back(edge.head);
        }
    };

    internal::read_metis(input, on_head, on_node, on_edge);
    first_out.emplace_back(head.size());
}

void gather_PE_ranges(NodeId local_from,
                      NodeId local_to,
                      std::vector<std::pair<NodeId, NodeId>>& ranges,
                      const MPI_Comm& comm) {
    MPI_Datatype MPI_RANGE;
    MPI_Type_vector(1, 2, 0, GRAPH_IO_MPI_NODE, &MPI_RANGE);
    MPI_Type_commit(&MPI_RANGE);
    std::pair<NodeId, NodeId> local_range(local_from, local_to);
    MPI_Allgather(&local_range, 1, MPI_RANGE, ranges.data(), 1, MPI_RANGE, comm);
#ifdef CHECK_RANGES
    int rank;
    MPI_Comm_rank(comm, &rank);
    if (rank == 0) {
        NodeId next_expected = 0;
        for (size_t i = 0; i < ranges.size(); ++i) {
            std::pair<NodeId, NodeId>& range = ranges[i];
            if (range.first == range.second) {
                continue;
            }
            if (range.first > range.second) {
                throw std::runtime_error("[R" + std::to_string(i) + "] range [" + std::to_string(range.first) + ", " +
                                         std::to_string(range.second) + "] is invalid");
            }
            if (range.first > next_expected) {
                throw std::runtime_error("[R" + std::to_string(i) + "] range [" + std::to_string(range.first) + ", " +
                                         std::to_string(range.second) + "] has a gap to previous one: [" +
                                         std::to_string(ranges[i - 1].first) + ", " +
                                         std::to_string(ranges[i - 1].second) + "]");
            }
            if (range.first < next_expected) {
                throw std::runtime_error("[R" + std::to_string(i) + "] range [" + std::to_string(range.first) + ", " +
                                         std::to_string(range.second) + "] overlaps with previous one: [" +
                                         std::to_string(ranges[i - 1].first) + ", " +
                                         std::to_string(ranges[i - 1].second) + "]");
            }
            next_expected = range.second;
        }
    }
#endif
    MPI_Type_free(&MPI_RANGE);
}

PEID get_PE_from_node_ranges(NodeId node, const std::vector<std::pair<NodeId, NodeId>>& ranges) {
    NodeId local_from;
    NodeId local_to;
    for (size_t i = 0; i < ranges.size(); ++i) {
        std::tie(local_from, local_to) = ranges[i];
        if (local_from <= node && node <= local_to) {
            return i;
        }
    }
    std::stringstream out;
    out << "Node " << node << " not assigned to any PE";
    throw std::runtime_error(out.str());
}

LocalGraphView read_local_metis_graph(const std::string& input, const GraphInfo& graph_info, PEID rank, PEID size) {
    std::vector<EdgeId> first_out;
    std::vector<NodeId> head;
    read_metis_distributed(input, graph_info, first_out, head, rank, size);

    std::vector<std::pair<NodeId, NodeId>> ranges(size);
    gather_PE_ranges(graph_info.local_from, graph_info.local_to, ranges, MPI_COMM_WORLD);

    std::vector<LocalGraphView::NodeInfo> node_info(first_out.size() - 1);
    for (size_t i = 0; i < first_out.size() - 1; ++i) {
        node_info[i].global_id = i + graph_info.local_from;
        node_info[i].degree = first_out[i + 1] - first_out[i];
    }
    return LocalGraphView{std::move(node_info), std::move(head)};
}

LocalGraphView read_local_partitioned_edgelist(const std::string& input, PEID rank, PEID size) {
    node_set ghosts;
    // ghosts.set_empty_key(-1);
    std::vector<Edge<>> edges;
    std::vector<std::pair<NodeId, NodeId>> ranges(size);

    auto input_path = std::filesystem::path(input + "_" + std::to_string(rank));

    NodeId local_from = std::numeric_limits<NodeId>::max();
    NodeId local_to = std::numeric_limits<NodeId>::min();
    internal::read_edge_list(
        input_path.string(),
        [&](Edge<> e) {
            if (e.tail > local_to) {
                local_to = e.tail;
            }
            if (e.tail < local_from) {
                local_from = e.tail;
            }
            edges.emplace_back(e);
        },
        1, "e", "c");
    local_to++;

    NodeId local_node_count = local_to - local_from;

    std::sort(edges.begin(), edges.end(), [&](const Edge<>& e1, const Edge<>& e2) {
        return std::tie(e1.tail, e1.head) < std::tie(e2.tail, e2.head);
    });

    // kagen sometimes produces duplicate edges
    auto it = std::unique(edges.begin(), edges.end());
    edges.erase(it, edges.end());

    NodeId total_node_count;
    MPI_Allreduce(&local_node_count, &total_node_count, 1, GRAPH_IO_MPI_NODE, MPI_SUM, MPI_COMM_WORLD);

    gather_PE_ranges(local_from, local_to, ranges, MPI_COMM_WORLD);

    fix_broken_edge_list(edges, ranges, ghosts, rank, size);

    NodeId current_node = std::numeric_limits<NodeId>::max();
    Degree degree_counter = 0;
    std::vector<LocalGraphView::NodeInfo> node_info;
    std::vector<NodeId> edge_heads;
    for (auto& edge : edges) {
        NodeId tail = edge.tail;
        NodeId head = edge.head;
        if (tail >= local_from && tail < local_to) {
            Edge<> e{tail, head};
            if (current_node != e.tail) {
                if (current_node != std::numeric_limits<NodeId>::max()) {
                    node_info.emplace_back(current_node, degree_counter);
                }
                degree_counter = 0;
            }
            edge_heads.emplace_back(e.head);
            degree_counter++;
        }
    }
    node_info.emplace_back(current_node, degree_counter);

    return LocalGraphView{std::move(node_info), std::move(edge_heads)};
}

LocalGraphView read_local_binary_graph(const std::string& input, const GraphInfo& graph_info, PEID rank, PEID size) {
    auto input_path = std::filesystem::path(input);
    auto basename = input_path.stem();
    auto path = input_path.parent_path();
    auto first_out_path = path / (basename.string() + ".first_out");
    auto head_path = path / (basename.string() + ".head");
    if (!std::filesystem::exists(first_out_path)) {
        throw std::runtime_error("File " + first_out_path.string() + " does not exist.");
    }
    if (!std::filesystem::exists(head_path)) {
        throw std::runtime_error("File " + head_path.string() + " does not exist.");
    }

    std::vector<EdgeId> first_out;
    std::vector<NodeId> head;
#if !defined(GRAPH_IO_MMAP)
    size_t first_index = graph_info.local_from * sizeof(EdgeId);
    {
        ConcurrentFile first_out_file(first_out_path.string(), ConcurrentFile::AccessMode::ReadOnly, MPI_COMM_WORLD);
        first_out_file.read(first_out, graph_info.local_node_count() + 1, first_index);
    }
    size_t to_read = first_out[graph_info.local_node_count()] - first_out[0];
    first_index = first_out[0] * sizeof(NodeId);
    {
        ConcurrentFile head_file(head_path.string(), ConcurrentFile::AccessMode::ReadOnly, MPI_COMM_WORLD);
        head_file.read(head, to_read, first_index);
    }
#else
    internal::read_binary<false>(input, first_out, head, graph_info.local_from, graph_info.local_node_count());
#endif
    std::vector<std::pair<NodeId, NodeId>> ranges(size);
    gather_PE_ranges(graph_info.local_from, graph_info.local_to, ranges, MPI_COMM_WORLD);

    std::vector<LocalGraphView::NodeInfo> node_info(first_out.size() - 1);
    for (size_t i = 0; i < first_out.size() - 1; ++i) {
        node_info[i].global_id = i + graph_info.local_from;
        node_info[i].degree = first_out[i + 1] - first_out[i];
    }
    return LocalGraphView{std::move(node_info), std::move(head)};
}

void read_graph_info_from_binary(const std::string& input, NodeId& node_count, EdgeId& edge_count) {
    auto input_path = std::filesystem::path(input);
    auto basename = input_path.stem();
    auto path = input_path.parent_path();
    auto first_out_path = path / (basename.string() + ".first_out");
    auto head_path = path / (basename.string() + ".head");
    if (!std::filesystem::exists(first_out_path)) {
        throw std::runtime_error("File " + first_out_path.string() + " does not exist.");
    }
    if (!std::filesystem::exists(head_path)) {
        throw std::runtime_error("File " + head_path.string() + " does not exist.");
    }
    {
        ConcurrentFile first_out_file(first_out_path.string(), ConcurrentFile::AccessMode::ReadOnly, MPI_COMM_WORLD);
        size_t file_size = first_out_file.size();
        if (file_size % sizeof(uint64_t) != 0) {
            throw std::runtime_error("Filesize is no multiple of 64 Bit");
        }
        node_count = file_size / sizeof(uint64_t) - 1;
    }
    {
        ConcurrentFile head_file(head_path.string(), ConcurrentFile::AccessMode::ReadOnly, MPI_COMM_WORLD);
        size_t file_size = head_file.size();
        if (file_size % sizeof(uint64_t) != 0) {
            throw std::runtime_error("Filesize is no multiple of 64 Bit");
        }
        edge_count = file_size / sizeof(uint64_t);
    }
}
std::pair<NodeId, NodeId> get_node_range(const std::string& input, PEID rank, PEID size) {
    (void)size;
    std::ifstream stream(input);
    if (stream.fail()) {
        throw std::runtime_error("Could not open input file for reading: " + input);
    }
    PEID pe = 0;
    std::string line;
    while (pe <= rank && std::getline(stream, line)) {
        pe++;
    }
    std::istringstream sstream(line);
    NodeId from, to;
    sstream >> pe >> from >> to;
    if (pe != rank) {
        throw std::runtime_error("Something went wrong");
    }
    return std::make_pair(from, to);
}
}  // namespace internal

std::vector<size_t> read_local_partition(const std::string& input, NodeId from, NodeId to, PEID rank, PEID size) {
    std::string partition_string;
    {
        ConcurrentFile partition_file(input, ConcurrentFile::AccessMode::ReadOnly, MPI_COMM_WORLD);
        std::vector<char> data;
        partition_file.read(data, partition_file.size() / sizeof(char));
        partition_string = std::string(data.begin(), data.end());
    }
    std::stringstream sstream(partition_string);
    std::string line;
    size_t line_index = 0;
    std::vector<size_t> partitioning;
    partitioning.reserve(to - from);
    while (std::getline(sstream, line, '\n')) {
        if (line_index < from) {
            line_index++;
            continue;
        }
        if (line_index >= to) {
            break;
        }
        std::stringstream linestream(line);
        size_t partition;
        linestream >> partition;
        partitioning.push_back(partition);
        if (linestream.bad()) {
            throw std::runtime_error("Failed to read partition");
        }
        line_index++;
    }
    return partitioning;
}

#ifdef GRAPH_IO_USE_KAGEN
IOResult gen_local_graph(const GeneratorParameters& conf_, PEID rank, PEID size) {
    GeneratorParameters conf = conf_;
    kagen::VertexRange vertex_range;

    kagen::SInt n = static_cast<kagen::SInt>(1) << conf.n;
    kagen::SInt m = static_cast<kagen::SInt>(1) << conf.m;

    kagen::KaGen gen(MPI_COMM_WORLD);
    if (conf.verify_graph) {
        gen.EnableUndirectedGraphVerification();
    }
    if (conf.statistics) {
        gen.EnableBasicStatistics();
    }
    gen.SetSeed(conf.seed);
    kagen::EdgeList edge_list;
    if (conf.generator == "gnm") {
        auto [edge_list_, vertex_range_] = gen.GenerateUndirectedGNM(n, m, false);
        edge_list = std::move(edge_list_);
        vertex_range = std::move(vertex_range_);
    } else if (conf.generator == "rdg_2d") {
        auto [edge_list_, vertex_range_] = gen.GenerateRDG2D(n, false);
        edge_list = std::move(edge_list_);
        vertex_range = std::move(vertex_range_);
    } else if (conf.generator == "rdg_3d") {
        auto [edge_list_, vertex_range_] = gen.GenerateRDG3D(n);
        edge_list = std::move(edge_list_);
        vertex_range = std::move(vertex_range_);
    } else if (conf.generator == "rgg_2d") {
        auto [edge_list_, vertex_range_] = gen.GenerateRGG2D_NM(n, m);
        edge_list = std::move(edge_list_);
        vertex_range = std::move(vertex_range_);
    } else if (conf.generator == "rgg_3d") {
        auto [edge_list_, vertex_range_] = gen.GenerateRGG3D_NM(n, m);
        edge_list = std::move(edge_list_);
        vertex_range = std::move(vertex_range_);
    } else if (conf.generator == "rhg") {
        auto [edge_list_, vertex_range_] = gen.GenerateRHG_NM(conf.gamma, n, m);
        edge_list = std::move(edge_list_);
        vertex_range = std::move(vertex_range_);
    } else if (conf.generator == "ba") {
        auto [edge_list_, vertex_range_] = gen.GenerateBA(n, conf.d);
        edge_list = std::move(edge_list_);
        vertex_range = std::move(vertex_range_);
    } else if (conf.generator == "grid_2d") {
        auto [edge_list_, vertex_range_] = gen.GenerateGrid2D_N(n, conf.p);
        edge_list = std::move(edge_list_);
        vertex_range = std::move(vertex_range_);
    } else if (conf.generator == "rmat") {
        auto [edge_list_, vertex_range_] = gen.GenerateRMAT(n, m, conf.a, conf.b, conf.c);
        edge_list = std::move(edge_list_);
        vertex_range = std::move(vertex_range_);
    } else {
        throw std::runtime_error("Generator not supported");
    }
    NodeId local_from = vertex_range.first;
    NodeId local_to = vertex_range.second;
    NodeId local_node_count = local_to - local_from;

    auto sort_by_tail = [](auto const& e1, auto const& e2) {
        return std::get<0>(e1) < std::get<0>(e2);
    };
    if (!std::is_sorted(edge_list.begin(), edge_list.end(), sort_by_tail)) {
        std::sort(edge_list.begin(), edge_list.end(), sort_by_tail);
    }

    if (edge_list.empty()) {
        internal::GraphInfo info;
        info.local_from = local_from;
        info.local_to = local_to;
        return {LocalGraphView(), info};
    }
    NodeId current_node = std::numeric_limits<NodeId>::max();
    Degree degree_counter = 0;
    NodeId node_counter = local_from;
    std::vector<LocalGraphView::NodeInfo> node_info;
    std::vector<NodeId> edge_heads;
    for (auto const& edge : edge_list) {
        NodeId tail = std::get<0>(edge);
        NodeId head = std::get<1>(edge);
        assert(tail >= local_from && tail < local_to);
        Edge<> e{tail, head};
        if (current_node != e.tail) {
            if (current_node != std::numeric_limits<NodeId>::max()) {
                node_info.emplace_back(current_node, degree_counter);
            }
            degree_counter = 0;
            current_node = e.tail;
        }
        edge_heads.emplace_back(e.head);
        degree_counter++;
    }
    node_info.emplace_back(current_node, degree_counter);

    internal::GraphInfo info;
    info.local_from = local_from;
    info.local_to = local_to;
    return {LocalGraphView{std::move(node_info), std::move(edge_heads)}, info};
}
#endif

IOResult read_local_graph(const std::string& input,
                          InputFormat format,
                          PEID rank,
                          PEID size,
                          bool degree_partitioned) {
    NodeId total_node_count;
    EdgeId total_edge_count;
    if (format == InputFormat::metis) {
        internal::read_metis_header(input, total_node_count, total_edge_count);
        total_edge_count *= 2;
    } else if (format == InputFormat::binary) {
        internal::read_graph_info_from_binary(input, total_node_count, total_edge_count);
    } else {
        throw std::runtime_error("Unsupported format");
    }
    internal::GraphInfo graph_info;
    if (degree_partitioned) {
        graph_info = internal::get_even_edge_distribution(input, format, rank, size);
    } else {
        graph_info = internal::GraphInfo::even_distribution(total_node_count, rank, size);
    }

    // atomic_debug("[" + std::to_string(graph_info.local_from) + ", " + std::to_string(graph_info.local_to) + ")");
    if (format == InputFormat::metis) {
        return {read_local_metis_graph(input, graph_info, rank, size), graph_info};
    } else if (format == InputFormat::binary) {
        return {read_local_binary_graph(input, graph_info, rank, size), graph_info};
    } else {
        throw std::runtime_error("This should not happen.");
    }
}

void write_graph_view(const LocalGraphView& G, const std::string& output, PEID rank, PEID size) {
    size_t nodes_to_write = G.node_info.size();
    size_t edges_to_write = G.edge_heads.size();
    std::array<size_t, 2> send_buf{nodes_to_write, edges_to_write};
    std::array<size_t, 2> recv_buf;
    MPI_Exscan(&send_buf, &recv_buf, 2, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    if (rank == 0) {
        recv_buf = {0, 0};
    }
    ConcurrentFile out_file(output, ConcurrentFile::AccessMode::ReadAndWrite | ConcurrentFile::AccessMode::Create,
                            MPI_COMM_WORLD);
    out_file.write_collective(std::vector{nodes_to_write, edges_to_write}, rank * 2 * sizeof(nodes_to_write));
    size_t offset = size * 2 * sizeof(nodes_to_write);
    offset += recv_buf[0] * sizeof(LocalGraphView::NodeInfo);
    offset += recv_buf[1] * sizeof(NodeId);
    out_file.write_collective(G.node_info, offset);
    offset += nodes_to_write * sizeof(LocalGraphView::NodeInfo);
    out_file.write_collective(G.edge_heads, offset);
}

IOResult read_graph_view(const std::string& input, PEID rank, PEID size) {
    ConcurrentFile in_file(input, ConcurrentFile::AccessMode::ReadOnly, MPI_COMM_WORLD);
    std::vector<std::pair<size_t, size_t>> local_sizes;
    in_file.read_collective(local_sizes, size);
    auto local_size = local_sizes[rank];
    exclusive_scan(local_sizes.begin(), local_sizes.end(), local_sizes.begin(), std::make_pair(size_t{0}, size_t{0}),
                   [](auto& lhs, auto& rhs) { return std::make_pair(lhs.first + rhs.first, lhs.second + rhs.second); });
    size_t offset = size * 2 * sizeof(size_t);
    offset += local_sizes[rank].first * sizeof(LocalGraphView::NodeInfo);
    offset += local_sizes[rank].second * sizeof(NodeId);
    LocalGraphView G;
    in_file.read_collective(G.node_info, local_size.first, offset);
    offset += local_size.first * sizeof(LocalGraphView::NodeInfo);
    in_file.read_collective(G.edge_heads, local_size.second, offset);
    internal::GraphInfo info;
    if (G.local_node_count() > 0) {
        info.local_from = G.node_info.front().global_id;
        info.local_to = G.node_info.back().global_id + 1;
    } else {
        info.local_from = std::numeric_limits<NodeId>::max();
        info.local_to = std::numeric_limits<NodeId>::max();
    }
    return {std::move(G), std::move(info)};
}

std::string dump_to_tmp(const LocalGraphView& G, PEID rank, PEID size) {
    std::string tmp_file = std::filesystem::temp_directory_path().string() + "/graphdumpXXXXXX";
    if (rank == 0) {
        int fd = mkstemp(tmp_file.data());
        close(fd);
    }
    MPI_Bcast(tmp_file.data(), tmp_file.size(), MPI_CHAR, 0, MPI_COMM_WORLD);
    write_graph_view(G, tmp_file, rank, size);
    return tmp_file;
}

}  // namespace graphio
