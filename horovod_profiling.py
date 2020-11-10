# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
# Modifications copyright (C) 2018 Uber Technologies, Inc.
# Modifications copyright (C) 2019 Intel Corporation
# Modifications copyright (C) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

"""Tests for horovod.tensorflow.mpi_ops."""

from distutils.version import LooseVersion
from google.protobuf import text_format
import itertools
import numpy as np
import os
import math
import tensorflow as tf
from tensorflow.python.framework import ops
import warnings
import time

import horovod.tensorflow as hvd



def test_horovod_allreduce_multi_gpu(enable_timeline=False, warmup=5, steps=20,
                save_profiling_graph=False, chain_len=10):
    """Test that the allreduce works on multiple GPUs.

    This test will crash badly if used with an MPI implementation that does
    not support GPU memory transfers directly, as it will call MPI_Send on
    a GPU data pointer."""
    # Only do this test if there are GPUs available.
    if not tf.test.is_gpu_available(cuda_only=True):
        print("[ERROR] No GPUs available")
        return -1

    # Only do this test if there are enough GPUs available.
    #if len(tf.config.experimental.list_physical_devices('GPU')) < 2:
    #    print("[ERROR] Too few GPUs available")
    #    return -1

    if int(os.environ.get('HOROVOD_MIXED_INSTALL', 0)):
        # Skip if compiled with CUDA but without HOROVOD_GPU_OPERATIONS.
        print("[ERROR] Not compiled with HOROVOD_GPU_OPERATIONS")
        return -1

    hvd.init()
    local_rank = hvd.local_rank()
    size = hvd.size()
    #dtypes = [tf.int32, tf.int64, tf.float16, tf.float32, tf.float64]
    #dims = [1, 2, 3]
    dtype = tf.float32
    shape_w = 1024
    #h_range = [1,2,4,8,16,32,64,128,256,512,1024,2048,4096] # 4KB - 16MB
    h_range = [4096]
    hooks=[]
    
    avg_time_list = []
    config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    #config.gpu_options.visible_device_list = str(hvd.local_rank())
    if hvd.local_rank()!=0:
        enable_timeline=False
    for shape_h in h_range:
        hvd_op_list = []
        tf.reset_default_graph()
        if enable_timeline:
            global_step = tf.train.get_or_create_global_step()
            timeline_hook = tf.train.ProfilerHook(save_steps=2,\
                            output_dir = 'exec_timeline')
            hooks.append(timeline_hook)
            increase_step_op = tf.assign_add(global_step, 1, name='inc_step')
        if save_profiling_graph == True:
            options = tf.RunOptions(output_partition_graphs=True)
            run_metadata = tf.RunMetadata()
        with tf.device("/gpu:%d" % local_rank):
            #tensor = tf.random.uniform(
            #    [shape_h, shape_w], 0.0, 1.0, dtype=dtype)
            data_a = tf.constant(1.0, shape=[shape_h, shape_w], dtype=dtype)
            data_b = tf.constant(2.0, shape=[shape_h, shape_w], dtype=dtype)
            tensor = data_a + data_b
            for i in range(chain_len):
                summed = hvd.allreduce(tensor, average=False)
                hvd_op_list.append(summed)
            final_op = [tf.shape_n(hvd_op_list)]
            if enable_timeline:
                final_op.append(increase_step_op)
        graph_def = tf.get_default_graph().as_graph_def()
        if hvd.local_rank() == 0:
            with open('basic_graph.pbtxt', 'w') as fdout:
                fdout.write(text_format.MessageToString(graph_def))
        with tf.train.MonitoredTrainingSession(config=config,hooks=hooks) as sess:
            avg_time_ms = 0.0
            for i in range(steps):
                start_t = time.time()
                if save_profiling_graph == True:
                    sess.run(final_op,options = options, run_metadata = run_metadata)
                else:
                    sess.run(final_op)
                end_t = time.time()
                duration_ms = (end_t - start_t)*1000.0
                print("Step %d: %.3f ms" % (i, duration_ms))
                if i >= warmup:
                    avg_time_ms += duration_ms
            avg_time_ms /= (steps - warmup)
            print("Avg: %.3f ms" % avg_time_ms)
            avg_time_list.append(avg_time_ms)

        if save_profiling_graph == True:
            print('Ready to dump partition graph')
            for i, partition_graph_def in enumerate(run_metadata.partition_graphs):
                meta_graph_path = './exec_graphs/'
                if not os.path.exists(meta_graph_path):
                    os.makedirs(meta_graph_path)
                with open('%s/%d.pbtxt' % (meta_graph_path,i), 'w') as f:
                    print(partition_graph_def, file=f)
        
        
        
    for shape_h in h_range:
        data_size = shape_h * shape_w * 4 * chain_len
        for avg_time_ms in avg_time_list:
            print("%d*%d*%d=%d Byte, time(ms): %.3f" % (shape_h, shape_w, chain_len, data_size, avg_time_ms))

if __name__ == '__main__':
    avg_time_list = test_horovod_allreduce_multi_gpu(enable_timeline=True,save_profiling_graph=True)
