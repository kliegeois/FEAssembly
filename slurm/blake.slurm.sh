#!/bin/bash
# Submission script for Blake
#SBATCH --time=47:00:00 # hh:mm:ss
#
#SBATCH -N 1
#SBATCH -p blake
#
#SBATCH --comment=test

export MKL_DYNAMIC=TRUE
export OMP_DYNAMIC=FALSE
export OMP_NUM_THREADS=24
export KMP_HW_SUBSET=24C,1T
export KMP_AFFINITY=compact
export OMP_PROC_BIND=true

export MKL_NUM_THREADS=${OMP_NUM_THREADS}

export base_dir="/ascldap/users/knliege/dev/FEAssembly/"

cd $base_dir
if [ ! -d "workspace" ]; then
  mkdir workspace
fi

cd workspace

mpirun -np 2  ./../../FEAssemblyB/src/FEAssembly --kokkos-threads=$OMP_NUM_THREADS
