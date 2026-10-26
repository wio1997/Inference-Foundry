# HCCL Test

HCCL Test provides HCCL communication performance and correctness testing.

## Environment Preparation

* Install CANN Toolkit Package

  Running HCCL Test tool requires CANN Toolkit development kit package and CANN operator package. Download the corresponding version of CANN software package according to operating system architecture. Refer to [Ascend Documentation Center - CANN Software Installation Guide](https://www.hiascend.com/document/redirect/CannCommunityInstWizard) for installation.

* Set CANN Software Environment Variables

  ```shell
  # Default path, root user installation
  source /usr/local/Ascend/cann/set_env.sh
  # Default path, non-root user installation
  source $HOME/Ascend/cann/set_env.sh
  ```

* Install and Configure MPI Environment Variables

  For MPI installation, refer to the "MPI Installation and Configuration" chapter in the [HCCL Performance Test Tool User Guide](https://www.hiascend.com/document/redirect/CannCommunityToolHcclTest).

  Environment variable configuration example:

  ```shell
  export PATH=/path/to/mpi/bin:$PATH
  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/mpi/lib
  ```

* For multi-machine cluster training, configure environment variable to specify host network card: (HCCL_SOCKET_IFNAME)

  ```shell
  # Configure HCCL initialization root communication network card name. HCCL can obtain Host IP through this network card name to complete communication domain creation.
  # Supported formats: (Choose one of the 4 specifications)
  # Exact match network card
  export HCCL_SOCKET_IFNAME==eth0,enp0     # Use specified eth0 or enp0 network card
  export HCCL_SOCKET_IFNAME=^=eth0,enp0    # Do not use eth0 and enp0 network cards
  # Fuzzy match network card
  export HCCL_SOCKET_IFNAME=eth,enp        # Use all network cards with eth or enp prefix
  export HCCL_SOCKET_IFNAME=^eth,enp       # Do not use any network cards with eth or enp prefix
  ```

  Note: Network card names are examples only and do not only apply to eth and enp network cards
  
* For multi-machine cluster training, collect host network card information used by all nodes

  Edit hostfile file:

  ```shell
  vim hostfile
  ```

  Write all participating training node information into hostfile file. The format is as follows:

  ```shell
  # All participating training node ip:processes per node
  192.168.1.1:8
  192.168.1.2:8
  ...
  ```

## Build

* Build HCCL Test

  Where MPI_HOME is the MPI installation path and ASCEND_DIR is the CANN Toolkit development kit package installation path.

  ```shell
  make MPI_HOME=/path/to/mpi ASCEND_DIR=${ASCEND_HOME_PATH}
  ```

## Execution

* Single node run:

  ```shell
  mpirun -n 8 ./bin/all_reduce_test -b 8K -e 64M -f 2 -d fp32 -o sum -p 8
  ```

* Multi-node run: (two nodes as example)

  ```shell
  mpirun -f hostfile -n 16 ./bin/all_reduce_test -b 8K -e 64M -f 2 -d fp32 -o sum -p 8
  ```

## Parameters

All tests support the same parameter set:

* NPU Count
  
  * `[-p,--npus <npus used for one node>]` Number of NPUs participating in training on each compute node. Default value: total NPUs on current node
  
* Data Volume
  * `-b,--minbytes <min size in bytes>` Data volume starting value. Default value: 64M
  * `-e,--maxbytes <max size in bytes>` Data volume ending value. Default value: 64M
  * Data increment through increment step or multiplication factor parameter setting
    * `-i,--stepbytes <increment size>` Increment step. Default value: (max-min)/10
    
      Note: When input increment step (-i) is 0, continuous testing on data volume starting value (-b) is performed.
    
    * `-f,--stepfactor <increment factor>` Multiplication factor. Default value: not enabled
  
* HCCL Operation Parameters
  * `-o,--op <sum/prod/max/min>` Collective communication operation reduction type. Default value: sum
  
  * `-r,--root <root>` Root node. Effective for broadcast, reduce, and scatter operations. Default value: 0
  
  * `-d,--datatype <int8/int16/int32/fp16/fp32/int64/uint64/uint8/uint16/uint32/fp64>` Data type. Default value: fp32 (that is, float32)

  * `-z,--zero_copy <0/1>` Enable zero copy. Effective for allgather, reduce_scatter, broadcast, allreduce operations that meet constraint conditions. Default value: 0
  * `-m,--symmetric_memory <0/1>` Enable symmetric memory. Effective for allgather, reduce_scatter, allreduce operations that meet constraint conditions. Default value: 0
      Note: Zero copy and symmetric memory cannot be enabled simultaneously.
  
* Performance
  * `-n,--iters <iteration count>` Iteration count. Default value: 20
  * `-w,--warmup_iters <warmup iteration count>` Warmup iteration count (not included in performance statistics, only affects HCCL Test execution time). Default value: 10
  * `-t,--onlydevicetime <0/1>` Exclude communication operator host-side software time and kernel loading time from communication execution time, only count device execution time (affects HCCL TEST execution time). Default value: 0
      Note:
      1. When -t parameter is enabled, -w and -n parameter configuration does not exceed 100 rounds
      2. When -t parameter is enabled, aicpu_ts mode is not supported
      3. When -t parameter is enabled and HCCL_BUFFSIZE configuration is less than or equal to 100MB, -t parameter does not take effect
* Result Verification
  
  * `-c,--check <0/1>` Verify collective communication operation result correctness (in large-scale cluster scenarios, enabling result verification increases HCCL Test execution time). Default value: 1 (enabled)

## Execution Examples

* allreduce

  ```shell
  # Single node 8 NPUs
  mpirun -n 8 ./bin/all_reduce_test -b 8K -e 64M -f 2 -p 8
  ```

  ```shell
  # Two nodes 16 NPUs
  mpirun -f hostfile -n 16 ./bin/all_reduce_test -b 8K -e 64M -f 2 -p 8
  ```

* broadcast

  ```shell
  # Single node 8 NPUs
  mpirun -n 8 ./bin/broadcast_test -b 8K -e 64M -f 2 -p 8 -r 1
  ```

  ```shell
  # Two nodes 16 NPUs
  mpirun -f hostfile -n 16 ./bin/broadcast_test -b 8K -e 64M -f 2 -p 8 -r 1
  ```

* allgather

  ```shell
  # Single node 8 NPUs
  mpirun -n 8 ./bin/all_gather_test -b 8K -e 64M -f 2 -p 8
  ```

  ```shell
  # Two nodes 16 NPUs
  mpirun -f hostfile -n 16 ./bin/all_gather_test -b 8K -e 64M -f 2 -p 8
  ```

* alltoallv

  ```shell
  # Single node 8 NPUs
  mpirun -n 8 ./bin/alltoallv_test -b 8K -e 64M -f 2 -p 8
  ```

  ```shell
  # Two nodes 16 NPUs
  mpirun -f hostfile -n 16 ./bin/alltoallv_test -b 8K -e 64M -f 2 -p 8
  ```

* alltoall

  ```shell
  # Single node 8 NPUs
  mpirun -n 8 ./bin/alltoall_test -b 8K -e 64M -f 2 -p 8
  ```

  ```shell
  # Two nodes 16 NPUs
  mpirun -f hostfile -n 16 ./bin/alltoall_test -b 8K -e 64M -f 2 -p 8
  ```

* reducescatter

  ```shell
  # Single node 8 NPUs
  mpirun -n 8 ./bin/reduce_scatter_test -b 8K -e 64M -f 2 -p 8
  ```

  ```shell
  # Two nodes 16 NPUs
  mpirun -f hostfile -n 16 ./bin/reduce_scatter_test -b 8K -e 64M -f 2 -p 8
  ```

* reduce

  ```shell
  # Single node 8 NPUs
  mpirun -n 8 ./bin/reduce_test -b 8K -e 64M -f 2 -p 8 -r 1
  ```
  
  ``` shell
  # Two nodes 16 NPUs
  mpirun -f hostfile -n 16 ./bin/reduce_test -b 8K -e 64M -f 2 -p 8 -r 1
  ```

## Run Cases with Specified deviceId

  Before executing HCCL Test tool, enable the following environment variable to specify the devices to start.

  Create an executable file in the hccl_test directory on the current compute node. For example: run.sh.

  * Single server scenario (start cards 4, 5, 6, 7)

    * Create executable file:

      run.sh file content:
      ```shell
      #Numbers after HCCL_TEST_USE_DEVS are the deviceIds to start
      export HCCL_TEST_USE_DEVS="4,5,6,7"
      $1
      ```
    * Case execution:

      ```shell
      mpirun -n 4 ./run.sh "./all_reduce_test -b 8K -e 64M -f 2 -p 4"
      ```

  * Multi-server scenario (compute node 1 starts cards 0, 1, 2, 3, compute node 2 starts cards 4, 5, 6, 7)

    * Compute node 1:

      * Create executable file:
      
        run.sh file content:
	    ```shell
        export HCCL_TEST_USE_DEVS="0,1,2,3"
        $1
	    ```
      
    * Compute node 2:

      * Create executable file:
      
        run.sh file content:
	    ```shell
        export HCCL_TEST_USE_DEVS="4,5,6,7"
        $1
	    ```

    * Case execution:

      ```shell
      mpirun -n 8 -f hostfile ./run.sh "./all_reduce_test -b 8K -e 64M -f 2 -p 4"
	  ```

## Enable Performance Data Collection

  Before executing HCCL Test tool, set the following environment variable to enable performance data collection.

  ```shell
  # "1" means enable profiling, "0" means disable profiling. Default value is "0". When enabled, performance data is collected during HCCL Test execution
  export HCCL_TEST_PROFILING=1
  # Specify profiling data storage path. Default is "/var/log/npu/profiling"
  export HCCL_TEST_PROFILING_PATH=/home/profiling
  ```
  After HCCL Test tool execution completes, profiling data is generated in the directory specified by HCCL_TEST_PROFILING_PATH.