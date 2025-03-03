#!/bin/bash

# SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# SCRIPT_DIR=$(cd "$(dirname "$(readlink -f "$0")")" && pwd)

# module unload compiler/dtk/24.04
# module load mpi/intelmpi/2021.3.0 
# module load compiler/intel/2021.3.0 compiler/cmake/3.23.3 compiler/dtk/23.10 
module load compiler/intel/2021.3.0 compiler/cmake/3.23.3

# cd /public/software/compiler/rocm/dtk-23.10/
cd /opt/dtk/
source env.sh
cd cuda/
source env.sh

cd $SCRIPT_DIR
