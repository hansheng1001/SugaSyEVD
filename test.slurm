#!/bin/bash
#SBATCH -J starccm+           # 作业的名称 可根据需要自行命名
#SBATCH -p xahcnormal         # 在指定分区中分配资源，根据所拥有的资源修改
#SBATCH -N 1                  # 申请的节点数1个
#SBATCH --ntasks-per-node=32  # 每个节点运行4个任务，使用4个核心

export DISPLAY=vadmin30:1

APP="/work/home/jsyadmin/apprepo/starccmplus/17.06.007-none/app/17.06.007-R8/STAR-CCM+17.06.007-R8/star/bin/starccm+ -power -mesa"

WDIR=`pwd`   #获取当前目录
cd $WDIR

NP=$SLURM_NPROCS
NNODE=`srun hostname |sort |uniq | wc -l`
LOG_FILE=$WDIR/job_${NP}c_${NNODE}n_$SLURM_JOB_ID.log
HOSTFILE=$WDIR/hosts_$SLURM_JOB_ID
srun hostname |sort |uniq -c |awk '{printf "%s:%s\n",$2,$1}' > $HOSTFILE

###############运行主程序
$APP  -machinefile $HOSTFILE -np $NP -rsh ssh  -mpidriver intel
