rlaunch  --charged-group=step1f --private-machine=group --gpu 8  --cpu 42 --memory 70600 \
--positive-tags H800,feature/gpfs=yes  --mount=juicefs+s3://oss.i.shaipower.com/xtan-jfs:/mnt/xtan-jfs \
--custom-resources rdma/mlnx_shared=8   -P 2 --set-env  DISTRIBUTED_JOB=true \
-- /data/Latte/models/parallel/torchrun.sh breakdown.py