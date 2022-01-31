#pragma once

#include <algorithm>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <ios>
#include <limits>
#include <sstream>
#include <vector>
#include "graph-io/graph_definitions.h"
#ifdef MPI_VERSION
#include "graph-io/mpi_io_wrapper.h"
#endif

namespace graphio {
namespace internal {

template <typename HeaderFunc, typename NodeFunc, typename EdgeFunc>
void read_metis(const std::string& input, HeaderFunc on_header, NodeFunc on_node, EdgeFunc on_edge) {
    std::ifstream stream(input);
    if (stream.fail()) {
        throw std::runtime_error("Could not open input file for reading: " + input);
    }
    std::string line;
    std::getline(stream, line);
    std::istringstream sstream(line);
    NodeId node_count;
    EdgeId edge_count;
    sstream >> node_count >> edge_count;
    if (sstream.bad()) {
        throw std::runtime_error("Failed to parse header.");
    }
    on_header(node_count, edge_count);

    size_t line_number = 1;
    NodeId node = 0;
    EdgeId edge_id = 0;
    while (node < node_count && std::getline(stream, line)) {
        sstream = std::istringstream(line);
        // skip comment lines
        if (line.rfind('%', 0) == 0) {
            line_number++;
            continue;
        }
        on_node(node);
        NodeId head_node;
        while (sstream >> head_node) {
            if (head_node >= node_count + 1) {
                throw std::runtime_error("Invalid node id " + std::to_string(head_node) + " in line " +
                                         std::to_string(line_number) + ".");
            }
            on_edge(Edge(node, head_node - 1));
            edge_id++;
        }
        if (sstream.bad()) {
            throw std::runtime_error("Invalid input in line " + std::to_string(line_number) + ".");
        }
        node++;
        line_number++;
    }
    if (node != node_count) {
        throw std::runtime_error("Number of nodes does not match header.");
    }
    if (edge_id != edge_count * 2) {
        std::stringstream out;
        out << "Number of edges does not mach header (header: " << edge_count << ", actual: " << edge_id << ")";
        throw std::runtime_error(out.str());
    }
}

inline void read_metis_header(const std::string& input, NodeId& node_count, EdgeId& edge_count) {
    std::ifstream stream(input);
    if (stream.fail()) {
        throw std::runtime_error("Could not open input file for reading: " + input);
    }
    std::string line;
    std::getline(stream, line);
    std::istringstream sstream(line);
    sstream >> node_count >> edge_count;
    if (sstream.bad()) {
        throw std::runtime_error("Failed to parse header.");
    }
}

template <typename EdgeFunc>
void read_edge_list(const std::string& input,
                    EdgeFunc on_edge,
                    NodeId starts_at = 0,
                    std::string edge_prefix = "",
                    std::string comment_prefix = "#") {
    std::ifstream stream(input);
    if (stream.fail()) {
        throw std::runtime_error("Could not open input file for reading: " + input);
    }

    std::string line;
    size_t line_number = 0;
    while (std::getline(stream, line)) {
        if (line.rfind(comment_prefix, 0) == 0) {
            line_number++;
            continue;
        }
        if (line.rfind(edge_prefix, 0) == 0) {
            line = line.substr(edge_prefix.size());
            std::istringstream sstream(line);
            NodeId tail;
            NodeId head;
            sstream >> tail >> head;
            if (sstream.bad()) {
                throw std::runtime_error("Invalid input in line " + std::to_string(line_number) + ".");
            }
            on_edge(Edge(tail - starts_at, head - starts_at));
        }
        ++line_number;
    }
}

template <bool use_mpi_io>
inline void read_binary(const std::string& input,
                        std::vector<EdgeId>& first_out,
                        std::vector<NodeId>& head,
                        NodeId first_node = 0,
                        NodeId node_count = std::numeric_limits<NodeId>::max()) {
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

    size_t first_index = first_node * sizeof(EdgeId);
    if constexpr (use_mpi_io) {
#ifdef MPI_VERSION
        ConcurrentFile first_out_file(first_out_path.string(), ConcurrentFile::AccessMode::ReadOnly, MPI_COMM_WORLD);
        size_t file_size = first_out_file.size();
        NodeId total_node_count = file_size / sizeof(EdgeId) - 1;
        NodeId nodes_to_read = std::min(node_count, total_node_count - first_index);
        first_out_file.read(first_out, nodes_to_read, first_index);
#else
        static_assert(!use_mpi_io, "MPI not enabled");
#endif
    } else {
        std::ifstream first_out_file(first_out_path, std::ios::binary);
        first_out_file.seekg(0, std::ios::end);
        size_t file_size = first_out_file.tellg();
        NodeId total_node_count = file_size / sizeof(EdgeId) - 1;
        NodeId nodes_to_read = std::min(node_count, total_node_count - first_index);
        first_out.resize(nodes_to_read + 1);
        size_t bytes_to_read = first_out.size() * sizeof(EdgeId);
        first_out_file.seekg(first_index, std::ios::beg);
        first_out_file.read(reinterpret_cast<char*>(first_out.data()), bytes_to_read);
    }
    size_t to_read = first_out[first_out.size() - 1] - first_out[0];
    first_index = first_out[0] * sizeof(NodeId);
    if constexpr (use_mpi_io) {
#ifdef MPI_VERSION
        ConcurrentFile head_file(head_path.string(), ConcurrentFile::AccessMode::ReadOnly, MPI_COMM_WORLD);
        head_file.read(head, to_read, first_index);
#endif
    } else {
        head.resize(to_read);
        std::ifstream head_file(head_path, std::ios::binary);
        size_t bytes_to_read = head.size() * sizeof(NodeId);
        head_file.seekg(first_index, std::ios::beg);
        head_file.read(reinterpret_cast<char*>(head.data()), bytes_to_read);
    }
}
}  // namespace internal
}  // namespace graphio
