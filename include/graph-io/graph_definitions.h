#pragma once

#include <cstdint>
#include <ostream>

namespace graphio {

using NodeId = std::uint64_t;
using EdgeId = std::uint64_t;
using Degree = NodeId;
#ifdef MPI_VERSION
#ifdef MPI_UNINT64_T
#define MPI_NODE MPI_UINT64_T
#else
static_assert(sizeof(unsigned long long) == 8, "We expect an unsigned long long to have 64 bit");
#define MPI_NODE MPI_UNSIGNED_LONG_LONG
#endif
#endif

struct Edge {
    Edge() : tail(0), head(0) {}
    Edge(NodeId tail, NodeId head) : tail(tail), head(head) {}
    Edge reverse() const {
        return Edge{head, tail};
    }
    template <typename VertexMap>
    Edge map(VertexMap map) {
        return Edge{map(tail), map(head)};
    }

    NodeId tail;
    NodeId head;
};

inline std::ostream& operator<<(std::ostream& out, const Edge& edge) {
    out << "(" << edge.tail << ", " << edge.head << ")";
    return out;
}

inline bool operator==(const Edge& x, const Edge& y) {
    return x.tail == y.tail && x.head == y.head;
}

}  // namespace graphio
