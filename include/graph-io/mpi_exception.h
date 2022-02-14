#pragma once

#include <mpi.h>
#include <array>
#include <sstream>
#include "graph-io/definitions.h"

namespace graphio {
struct MPIException : public std::exception {
    MPIException(const std::string& msg) : msg_() {
        PEID rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        std::stringstream out;
        out << "[R" << rank << "] " << msg;
        msg_ = out.str();
    }
    const char* what() const throw() {
        return msg_.c_str();
    }

private:
    std::string msg_;
};

inline void check_mpi_error(int errcode, const std::string& file, int line) {
    if (errcode != MPI_SUCCESS) {
        std::array<char, MPI_MAX_ERROR_STRING> buf;
        int resultlen;
        MPI_Error_string(errcode, buf.data(), &resultlen);
        std::string msg(buf.begin(), buf.begin() + resultlen);
        msg = msg + " in " + file + ":" + std::to_string(line);
        throw MPIException(msg);
    }
}
}  // namespace graphio

