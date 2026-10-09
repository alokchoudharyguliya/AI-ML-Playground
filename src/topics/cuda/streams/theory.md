## One queue is in order. Two queues can overlap

A **stream** is a queue of GPU work. Inside one stream, the next operation waits for the previous one: a kernel does not start until that stream's copy has finished, and a copy back does not start until the kernel has. Across streams there is no such rule. The GPU runs whatever is ready, on whatever hardware is free.

That hardware is not one pipe. A data-centre GPU has **two copy engines** (one host-to-device, one device-to-host) and the SMs. Those three can be busy at the same time. A consumer GPU may have a single copy engine, in which case the two directions take turns and only the kernel overlaps a copy.

```cuda
cudaMemcpyAsync(d, h + c * n, bytes, cudaMemcpyHostToDevice, stream);
kernel<<<grid, block, 0, stream>>>(d, n);          // the 0 is shared memory, then the stream
cudaMemcpyAsync(h + c * n, d, bytes, cudaMemcpyDeviceToHost, stream);
```

The host enqueues all of this and continues. It does not wait per chunk. One `cudaDeviceSynchronize()` (or an event) at the end is the whole pipeline.

## Pinned memory is what makes the copy asynchronous

`cudaMemcpyAsync` from ordinary `malloc` memory is allowed to block. The driver has to stage pageable pages into a pinned buffer before the DMA can start, and that staging runs on the host. The call returns late, the kernel you hoped to overlap is already finished, and extra streams change nothing.

**Pinned** (page-locked) memory is allocated with `cudaMallocHost` / `cudaHostAlloc` and released with `cudaFreeHost`. The GPU's DMA reads it directly. The async copy returns immediately, and the copy engine runs it beside whatever kernel is already on the SMs.

Pin the buffers you reuse. Pinning is slow, and pinned pages are a scarce OS resource. Pinning every temporary is worse than not overlapping.

## What does not overlap

- **The same stream.** Order is the point of a stream.
- **The legacy default stream** (`stream` argument 0, or a launch with no stream). It waits for every blocking stream, then blocks them. A copy launched with no stream beside a kernel you carefully put in a stream still will not overlap.
- **A blocking stream and the default stream.** `cudaStreamCreate` makes a blocking stream. `cudaStreamCreateWithFlags(s, cudaStreamNonBlocking)` does not synchronize with the legacy default stream. The compile flag `--default-stream per-thread` gives each host thread its own non-blocking default stream.

Two kernels overlap only when they are in different non-conflicting streams **and** both fit on the SMs at once. A kernel that already fills the GPU, which is the usual case after the occupancy chapter, does not share the SMs. The overlap this chapter is about is copy against compute.

## Events, not device-wide stalls

`cudaDeviceSynchronize()` waits for every stream. That is the right call once, at the end. It is the wrong call between chunks: it drains the pipeline you just built.

A cross-stream dependency is an event:

```cuda
cudaEventRecord(ready, streamA);            // when A reaches here
cudaStreamWaitEvent(streamB, ready, 0);     // B waits; the host does not
```

The host keeps enqueuing. Only stream B's later work is held. `cudaEventSynchronize` is the version that does stop the host, used for timing.

## How many streams

Three stages (copy in, kernel, copy out) need **three streams** to keep all three units busy when the stages are similar lengths. A fourth stream has no fourth unit to fill, so the timeline stops getting shorter. When one stage dominates — a huge copy, or a kernel much longer than the copies — two streams already hit the bottleneck, and more streams copy the same engine's queue.

The lab is that timeline. `pipeline.cu` is the same loop on a real device.
