# Horovod Profiling

## How to run

```bash
CUDA_VISIBLE_DEVICES=2,3 horovodrun -np 2 -H localhost:2 python horovod_profiling.py
```