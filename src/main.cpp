#include "Teuchos_GlobalMPISession.hpp"
#include "Teuchos_oblackholestream.hpp"
#include "Teuchos_XMLParameterListCoreHelpers.hpp"

// Tpetra
#include "Stokhos_Tpetra_MP_Vector.hpp"
#include "Stokhos_Tpetra_Utilities_MP_Vector.hpp"
#include "Tpetra_Core.hpp"
#include "Tpetra_Map.hpp"
#include "Tpetra_MultiVector.hpp"
#include "Tpetra_Vector.hpp"
#include "Tpetra_CrsGraph.hpp"
#include "Tpetra_CrsMatrix.hpp"
#include "Stokhos_Tpetra_CG.hpp"

#include <MatrixMarket_Tpetra.hpp>

#include <math.h>
#include <mpi.h>

#include <cstring>
#include <sstream>
#include <iostream>
#include <string>

#include "Stokhos_Ifpack2_MP_Vector.hpp"
#include "Ifpack2_Factory.hpp"

using std::string;

template <typename scalar, typename LO, typename GO, typename Node>
void example_1D(int argc, char **argv, GO n)
{
    typedef Tpetra::Map<LO, GO, Node> map_type;
    typedef Tpetra::MultiVector<scalar, LO, GO, Node> multivector_type;

    typedef Tpetra::Vector<scalar, LO, GO, Node> vector_type;
    typedef Tpetra::CrsGraph<LO, GO, Node> crs_graph_type;
    typedef Tpetra::CrsMatrix<scalar, LO, GO, Node> crs_matrix_type;
    typedef KokkosSparse::CrsMatrix<scalar, LO, typename crs_matrix_type::execution_space> local_matrix_type;
    typedef Tpetra::Export<LO, GO, Node> export_type;

    using Teuchos::RCP;
    using Teuchos::rcp;
    using Teuchos::tuple;

    RCP<const Teuchos::Comm<int>> comm = rcp(new Teuchos::MpiComm<int>(MPI_COMM_WORLD));

    const size_t myrank = comm->getRank();
    const size_t worldsize = comm->getSize();

    // Make an output stream (for verbose output) that only prints on
    // Proc 0 of the communicator.
    Teuchos::oblackholestream blackHole;
    std::ostream &out = (myrank == 0) ? std::cout : blackHole;

    double L = 1.;

    Kokkos::initialize(argc, argv);
    {
        const GO indexBase = 0;

        std::vector<GO> myNodesWO = {};

        const int n_per_process = ceil(n / worldsize);

        const GO my_first_element_index = myrank * n_per_process;
        const GO my_last_element_index = (myrank == worldsize - 1) ? n : (myrank + 1) * n_per_process;

        const GO my_first_node_index = my_first_element_index;
        const GO my_last_node_index = my_last_element_index + 1;

        for (GO i = my_first_node_index; i < my_last_node_index; ++i)
            myNodesWO.push_back(i);

        RCP<const map_type> mapNodesWO = rcp(new map_type(Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid(), myNodesWO, indexBase, comm));
        RCP<const map_type> mapNodes = Tpetra::createOneToOne(mapNodesWO);

        out << *mapNodesWO << std::endl;

        export_type exportAWO(mapNodesWO, mapNodes);

        out << "Maps have been created" << std::endl;

        RCP<crs_graph_type> graphAWO(new crs_graph_type(mapNodesWO, 0));
        for (GO i = my_first_element_index; i < my_last_element_index; ++i)
        {
            // Element i connects node i and node i + 1
            graphAWO->insertGlobalIndices(i, tuple<GO>(i, i + 1));
            graphAWO->insertGlobalIndices(i + 1, tuple<GO>(i, i + 1));
        }
        graphAWO->fillComplete();

        out << *graphAWO << std::endl;

        out << "Graph with overlaps has been created" << std::endl;

        RCP<crs_graph_type> graphA(new crs_graph_type(mapNodes, 0));
        graphA->doExport(*graphAWO, exportAWO, Tpetra::INSERT);
        graphA->fillComplete();

        out << "Graph without overlaps has been created" << std::endl;

        const scalar h = L / (n + 1);

        RCP<crs_matrix_type> AWO(new crs_matrix_type(graphAWO));

        AWO->setAllToScalar((scalar)0.0);
        AWO->fillComplete();
        AWO->resumeFill();        

        scalar vals[4] = {1. / (h * h), -1. / (h * h), -1. / (h * h), 1. / (h * h)};

        
        local_matrix_type local_AWO = AWO->getLocalMatrix();
        Kokkos::parallel_for(my_last_element_index - my_first_element_index, KOKKOS_LAMBDA(const GO i) {
            LO cols[2] = {i, i + 1};

            for (size_t j = 0; j < 2; ++j)
            {
                local_AWO.sumIntoValues(i + j, cols, 2, &vals[j * 2], false, true);
            }
        });
        
        /*
        Tpetra::MatrixMarket::Writer<crs_matrix_type>::writeSparseFile(std::string("AWO_0.txt"), AWO);

        for (GO i = my_first_element_index; i < my_last_element_index; ++i)
        {
            AWO->sumIntoGlobalValues(i, tuple<GO>(i, i + 1), tuple<scalar>(scalar(1.) / (h * h), scalar(-1.) / (h * h)));
            AWO->sumIntoGlobalValues(i+1, tuple<GO>(i, i + 1), tuple<scalar>(scalar(-1.) / (h * h), scalar(1.) / (h * h)));
        }
        */
        AWO->fillComplete();

        out << "Matrix with overlaps has been created" << std::endl;

        RCP<crs_matrix_type> A(new crs_matrix_type(graphA));
        A->doExport(*AWO, exportAWO, Tpetra::ADD);
        A->fillComplete();

        out << "Matrix without overlaps has been created" << std::endl;

        Tpetra::MatrixMarket::Writer<crs_matrix_type>::writeSparseFile(std::string("A.txt"), A);

        out << "Matrix without overlaps has been written on disk" << std::endl;
    }
    Kokkos::finalize();
}

int main(int argc, char **argv)
{
    Teuchos::GlobalMPISession session(&argc, &argv, NULL);

    int n = 1000;
    example_1D<double, int, int, typename Tpetra::Vector<>::node_type>(argc, argv, n);
}
