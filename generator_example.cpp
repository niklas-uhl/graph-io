#include <mpi.h>
#include <iostream>
#include "graph-io/graph_io.h"

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    graphio::GeneratorParameters params;
    params.generator = "rgg_2d";
    params.r_coeff = 0.55;
    params.n = 10;
    auto result = graphio::gen_local_graph(params, rank, size);
    auto G = std::move(result.G);
    std::cout << "n=" << G.local_node_count() << std::endl;
    std::cout << "m=" << G.edge_heads.size() << std::endl;

    return MPI_Finalize();
}
