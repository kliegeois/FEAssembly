FILE(GLOB SRCS *.h *.cpp *.inl *.swg)
FILE(GLOB ISRCS *.i)

SET(CMAKE_SWIG_FLAGS "")
SET_SOURCE_FILES_PROPERTIES(${ISRCS} PROPERTIES CPLUSPLUS ON)

ADD_EXECUTABLE(FEAssembly ${SRCS})

# -- Search for Trilinos
FIND_PACKAGE(Trilinos REQUIRED)

set (_VERB 0)  # set to 1 for debugging
IF(_VERB)
    MESSAGE("\nFound Trilinos!  Here are the details: ")
    MESSAGE("   Trilinos_DIR = ${Trilinos_DIR}")
    MESSAGE("   Trilinos_VERSION = ${Trilinos_VERSION}")
    MESSAGE("   Trilinos_PACKAGE_LIST = ${Trilinos_PACKAGE_LIST}")
    MESSAGE("   Trilinos_LIBRARIES = ${Trilinos_LIBRARIES}")
    MESSAGE("   Trilinos_INCLUDE_DIRS = ${Trilinos_INCLUDE_DIRS}")
    MESSAGE("   Trilinos_LIBRARY_DIRS = ${Trilinos_LIBRARY_DIRS}")
    MESSAGE("   Trilinos_TPL_LIST = ${Trilinos_TPL_LIST}")
    MESSAGE("   Trilinos_TPL_INCLUDE_DIRS = ${Trilinos_TPL_INCLUDE_DIRS}")
    MESSAGE("   Trilinos_TPL_LIBRARIES = ${Trilinos_TPL_LIBRARIES}")
    MESSAGE("   Trilinos_TPL_LIBRARY_DIRS = ${Trilinos_TPL_LIBRARY_DIRS}")
    MESSAGE("   Trilinos_BUILD_SHARED_LIBS = ${Trilinos_BUILD_SHARED_LIBS}")
    MESSAGE("End of Trilinos details\n")
ENDIF()

# MPI check 
LIST(FIND Trilinos_TPL_LIST MPI MPI_List_ID)
IF (MPI_List_ID GREATER -1)
  MESSAGE("-- Checking if MPI is enabled in Trilinos: MPI ENABLED")
  SET(MYAPP_MPI TRUE)
  ADD_DEFINITIONS(-DMYAPP_MPI)
ELSE()
  MESSAGE("-- Checking if MPI is enabled in Trilinos: MPI NOT ENABLED")
  SET(MYAPP_MPI FALSE)
ENDIF()

# Set optional dependency in MyApp on Epetra package: 
#   this toggles code within  #ifdef MYAPP_EPETRA 
LIST(FIND Trilinos_PACKAGE_LIST Epetra Epetra_List_ID)
IF (Epetra_List_ID GREATER -1)
  ADD_DEFINITIONS(-DMYAPP_EPETRA)
  MESSAGE("-- Looking for Epetra: -- found, compiling with -DMYAPP_EPETRA")
  SET(MYAPP_EPETRA TRUE)
ELSE()
  MESSAGE("-- Looking for Epetra: -- not found.")
  SET(MYAPP_EPETRA FALSE)
ENDIF()

# --------------------------------------------------------------
# Finds MPI (including MSMPI)
# --------------------------------------------------------------
if(WIN32)  
    FIND_PATH(MPI_INCLUDE_PATH NAMES  mpi.h HINTS "$ENV{MSMPI_INC}")
    if("${CMAKE_SIZEOF_VOID_P}" EQUAL 8)
        set(MS_MPI_ARCH_DIR x64)
        find_library(MPI_LIBRARIES msmpi PATHS "$ENV{MSMPI_LIB64}")
    else()
        set(MS_MPI_ARCH_DIR x86)
        find_library(MPI_LIBRARIES msmpi PATHS "$ENV{MSMPI_LIB32}")
    endif()
else()
    find_package(MPI REQUIRED)
    ADD_DEFINITIONS(-DMPICH_IGNORE_CXX_SEEK -DMPICH_SKIP_MPICXX)
endif()
# --------------------------------------------------------------


IF(_VERB)
MESSAGE("   Trilinos_LIBRARIES = ${Trilinos_LIBRARIES}")
ENDIF()

list(REMOVE_ITEM Trilinos_LIBRARIES pytrilinos)

IF(_VERB)
MESSAGE("   Trilinos_LIBRARIES = ${Trilinos_LIBRARIES}")
ENDIF()
# ---

INCLUDE_DIRECTORIES( ${MPI_INCLUDE_PATH}
                     ${Trilinos_INCLUDE_DIRS}
                     ${Trilinos_TPL_INCLUDE_DIRS}
                     )

LINK_DIRECTORIES (${Trilinos_LIBRARY_DIRS} ${Trilinos_TPL_LIBRARY_DIRS})

TARGET_LINK_LIBRARIES(  FEAssembly
                        ${Trilinos_LIBRARIES}
                        ${Trilinos_TPL_LIBRARIES}
                        ${MPI_LIBRARIES})



