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
    /*
        This example illustrates the matrix assembly process of 1D Laplacian problem
        using the finite element method and hybrid parallalism.

        The presented implementation does not use ghost cells and relies on the 
        summation of the contribution at the interface.

        This function has 4 template parameters:
        - scalar: the type of data stored in the vectors and in the matrix,
        - LO: the local ordinal type, the integer type used to store the local indices,
        - GO: the global ordinal type, the integer type used to store the global indices,
        - Node: the node type: the node-level parallel programming model used by Tpetra.
    */

    // First, we define some types using typedef to shorten the remaining code:
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

    // The first object created is the MPI communicator:
    RCP<const Teuchos::Comm<int>> comm = rcp(new Teuchos::MpiComm<int>(MPI_COMM_WORLD));

    // Using this communicator, we can access the rank of the current process:
    const size_t myrank = comm->getRank();
    // and the number of processes used:
    const size_t worldsize = comm->getSize();

    // After that, we make an output stream (for verbose output) that only
    // prints on the first MPI process (rank 0) of the communicator.
    Teuchos::oblackholestream blackHole;
    std::ostream &out = (myrank == 0) ? std::cout : blackHole;

    // We set the length of the 1D domain to 1:
    double L = 1.;

    // Now, we initialize Kokkos with argc and argv.
    // Doing so, the number of used threads has to be specified as an argument
    // of the executable.
    Kokkos::initialize(argc, argv);
    {
        /*
            -----------------------
            --  Map computation  --
            -----------------------
        */

        // The first step is to compute the Map, i.e. how the Degrees of
        // Freedom (DoF) of the problem are distributed over the MPI processes.

        // Here, we chose to use a zero-based indexing for the Map, Vector,
        // and Matrices:
        const GO indexBase = 0;

        // To create the Map, each MPI process will fill a vector of GO with
        // the indices of the DoFs which are associated with elements owned by
        // the calling MPI process.
        std::vector<GO> myNodesWO = {};

        // We have chosen to distribute the elements evenly: each process has
        // (at most) n_per_process elements:
        const int n_per_process = ceil(n / worldsize);
        // where n is the total number of elements.

        // Moreover, we have chosen to distribute the elements such that the
        // the first set of n_per_process elements is associated to the rank 0,
        // the second set of n_per_process elements is associated to the rank 1,
        // ...
        // To do so, we use 2 variables my_first_element_index and my_last_element_index
        // which represent the index of first and the index of the last element + 1 associated
        // to the calling MPI process respectively:
        const GO my_first_element_index = myrank * n_per_process;
        const GO my_last_element_index = (myrank == worldsize - 1) ? n : (myrank + 1) * n_per_process;
        // If the calling MPI process is the last one, the last index is n.

        // Based on those two element indices, we compute the DoFs indices:
        const GO my_first_node_index = my_first_element_index;
        const GO my_last_node_index = my_last_element_index + 1;

        // And we can fill the vector with all the indices between the bounds
        // my_first_node_index (included) and my_last_node_index (not included):
        for (GO i = my_first_node_index; i < my_last_node_index; ++i)
            myNodesWO.push_back(i);

        // With those vectors, we can initialize a map with overlaps:
        RCP<const map_type> mapNodesWO = rcp(new map_type(Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid(), myNodesWO, indexBase, comm));
        // And deduce a 1-to-1 map:
        RCP<const map_type> mapNodes = Tpetra::createOneToOne(mapNodesWO);

        out << *mapNodesWO << std::endl;

        // Based on the map with and without overlap, we can create
        // an export object:
        export_type exportAWO(mapNodesWO, mapNodes);

        out << "Maps have been created" << std::endl;

        /*
            -----------------------
            -- Graph computation --
            -----------------------
        */

        // The second main step is to compute the Graph, i.e.
        // the sparsity pattern of the matrix of the linear
        // system considered.

        // Here, we consider a 1D Laplacian problem, the sparsity
        // pattern is the one of a tridiagonal matrix.

        // However, we will create this pattern using a more general way.
        // We loop over each element owned and insert the connections
        // one by one.
        
        // First, we create a graph with overlaps based on the Map with
        // overlaps:
        RCP<crs_graph_type> graphAWO(new crs_graph_type(mapNodesWO, 0));

        // Then, we loop over the elements owned and insert the connections.
        for (GO i = my_first_element_index; i < my_last_element_index; ++i)
        {
            // Element i connects node i and node i + 1
            graphAWO->insertGlobalIndices(i, tuple<GO>(i, i + 1));
            graphAWO->insertGlobalIndices(i + 1, tuple<GO>(i, i + 1));
        }
        // Finally, we specify that we are done setting the connections.
        graphAWO->fillComplete();

        out << *graphAWO << std::endl;

        out << "Graph with overlaps has been created" << std::endl;

        // Now, we must create the sparsity pattern of the matrix without.
        // To do so, we start from the node map without overlaps:
        RCP<crs_graph_type> graphA(new crs_graph_type(mapNodes, 0));
        // And we export the previous graph with overlaps to the current one:
        graphA->doExport(*graphAWO, exportAWO, Tpetra::INSERT);
        graphA->fillComplete();

        out << "Graph without overlaps has been created" << std::endl;

        /*
            -----------------------
            -- Matrix computation--
            -----------------------
        */

        // Finally, we compute the entries of the matrix following the same
        // approach as for the Graph: we start from the matrix with overlaps
        // to deduce the matrix without overlaps.

        // First, we compute the mesh element size:
        const scalar h = L / n;

        // We allocate the matrix with overlaps knowning its graph:
        RCP<crs_matrix_type> AWO(new crs_matrix_type(graphAWO));

        // We set all the entries to zero:
        AWO->setAllToScalar((scalar)0.0);

        scalar vals[4] = {1. / (h * h), -1. / (h * h), -1. / (h * h), 1. / (h * h)};

        // And we loop over the owned elements and we sum their contributions:
        local_matrix_type local_AWO = AWO->getLocalMatrix();
        Kokkos::parallel_for(
            my_last_element_index - my_first_element_index, KOKKOS_LAMBDA(const GO i) {
                LO cols[2] = {i, i + 1};

                for (size_t j = 0; j < 2; ++j)
                {
                    local_AWO.sumIntoValues(i + j, cols, 2, &vals[j * 2], false, true);
                }
            });

        // Finally, we say that the assembly of the matrix with overlaps is done.
        AWO->fillComplete();

        out << "Matrix with overlaps has been created" << std::endl;

        // As in the case of the Graph, we use the export object to deduce
        // the matrix without overlaps:
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
