#pragma once

#include <cstdint>
#include <ostream>
#ifdef GRAPH_IO_PARALLEL
#include <mpi.h>
#ifdef MPI_UINT64_T
#define GRAPH_IO_MPI_NODE MPI_UINT64_T
#else
static_assert(sizeof(unsigned long long) == 8, "We expect an unsigned long long to have 64 bit");
#define GRAPH_IO_MPI_NODE MPI_UNSIGNED_LONG_LONG
#endif
#endif

namespace graphio {

using NodeId = std::uint64_t;
using EdgeId = std::uint64_t;
using Degree = NodeId;

template <typename NodeIdType = NodeId>
struct Edge {
    Edge() : tail(), head() {}
    Edge(NodeIdType tail, NodeIdType head) : tail(tail), head(head) {}
    Edge reverse() const {
        return Edge{head, tail};
    }
    template <typename VertexMap>
    Edge map(VertexMap map) {
        return Edge{map(tail), map(head)};
    }

    NodeIdType tail;
    NodeIdType head;
};

template <typename NodeIdType>
inline std::ostream& operator<<(std::ostream& out, const Edge<NodeIdType>& edge) {
    out << "(" << edge.tail << ", " << edge.head << ")";
    return out;
}

template <typename NodeIdType>
inline bool operator==(const Edge<NodeIdType>& x, const Edge<NodeIdType>& y) {
    return x.tail == y.tail && x.head == y.head;
}

}  // namespace graphio
