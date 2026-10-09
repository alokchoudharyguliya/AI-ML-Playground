import{r as e}from"./rolldown-runtime-hePW80VL.js";import{b as t,v as n}from"./r3f-x1z21uF6.js";import{g as r,t as i,y as a}from"./viz-CPys2405.js";import{t as o}from"./Stage3D-CWOT2b2l.js";import{a as s,c,i as l,l as u,o as d,r as f,s as p,t as m,u as h}from"./hooks-Dw7oo1m3.js";var g=e(t(),1);function _({n:e,streams:t,h:n,k:r,d:i,pinned:a,splitEngines:o}){let s=[],c=0;for(let t=0;t<e;t++)s.push({i:t,kind:`h`,stream:0,start:c,end:c+=n}),s.push({i:t,kind:`k`,stream:0,start:c,end:c+=r}),s.push({i:t,kind:`d`,stream:0,start:c,end:c+=i});let l=c;if(!a||t<1)return{ops:s,total:l,serialTotal:l};let u=Array.from({length:t},()=>[]);for(let n=0;n<e;n++)u[n%t].push(n);let d=Array(t).fill(0),f=Array(t).fill(0),p=Array(t).fill(0),m={h:0,k:0,d:0},h=[];for(let a=0;a<e*3;a++){let e=null;for(let n=0;n<t;n++){if(d[n]>=u[n].length)continue;let t=`hkd`[f[n]],r=t===`d`&&!o?`h`:t,i=Math.max(p[n],m[r]),a=u[n][d[n]];(!e||i<e.start||i===e.start&&a<e.i)&&(e={s:n,kind:t,resource:r,start:i,i:a})}let a=e.kind===`h`?n:e.kind===`k`?r:i,s=e.start+a;h.push({i:u[e.s][d[e.s]],kind:e.kind,stream:e.s,start:e.start,end:s}),p[e.s]=s,m[e.resource]=s,f[e.s]+=1,f[e.s]===3&&(f[e.s]=0,d[e.s]+=1)}return{ops:h,total:Math.max(...h.map(e=>e.end)),serialTotal:l}}function v(e,t){return e.filter(e=>e.start<=t&&t<e.end)}var y=`## One queue is in order. Two queues can overlap

A **stream** is a queue of GPU work. Inside one stream, the next operation waits for the previous one: a kernel does not start until that stream's copy has finished, and a copy back does not start until the kernel has. Across streams there is no such rule. The GPU runs whatever is ready, on whatever hardware is free.

That hardware is not one pipe. A data-centre GPU has **two copy engines** (one host-to-device, one device-to-host) and the SMs. Those three can be busy at the same time. A consumer GPU may have a single copy engine, in which case the two directions take turns and only the kernel overlaps a copy.

\`\`\`cuda
cudaMemcpyAsync(d, h + c * n, bytes, cudaMemcpyHostToDevice, stream);
kernel<<<grid, block, 0, stream>>>(d, n);          // the 0 is shared memory, then the stream
cudaMemcpyAsync(h + c * n, d, bytes, cudaMemcpyDeviceToHost, stream);
\`\`\`

The host enqueues all of this and continues. It does not wait per chunk. One \`cudaDeviceSynchronize()\` (or an event) at the end is the whole pipeline.

## Pinned memory is what makes the copy asynchronous

\`cudaMemcpyAsync\` from ordinary \`malloc\` memory is allowed to block. The driver has to stage pageable pages into a pinned buffer before the DMA can start, and that staging runs on the host. The call returns late, the kernel you hoped to overlap is already finished, and extra streams change nothing.

**Pinned** (page-locked) memory is allocated with \`cudaMallocHost\` / \`cudaHostAlloc\` and released with \`cudaFreeHost\`. The GPU's DMA reads it directly. The async copy returns immediately, and the copy engine runs it beside whatever kernel is already on the SMs.

Pin the buffers you reuse. Pinning is slow, and pinned pages are a scarce OS resource. Pinning every temporary is worse than not overlapping.

## What does not overlap

- **The same stream.** Order is the point of a stream.
- **The legacy default stream** (\`stream\` argument 0, or a launch with no stream). It waits for every blocking stream, then blocks them. A copy launched with no stream beside a kernel you carefully put in a stream still will not overlap.
- **A blocking stream and the default stream.** \`cudaStreamCreate\` makes a blocking stream. \`cudaStreamCreateWithFlags(s, cudaStreamNonBlocking)\` does not synchronize with the legacy default stream. The compile flag \`--default-stream per-thread\` gives each host thread its own non-blocking default stream.

Two kernels overlap only when they are in different non-conflicting streams **and** both fit on the SMs at once. A kernel that already fills the GPU, which is the usual case after the occupancy chapter, does not share the SMs. The overlap this chapter is about is copy against compute.

## Events, not device-wide stalls

\`cudaDeviceSynchronize()\` waits for every stream. That is the right call once, at the end. It is the wrong call between chunks: it drains the pipeline you just built.

A cross-stream dependency is an event:

\`\`\`cuda
cudaEventRecord(ready, streamA);            // when A reaches here
cudaStreamWaitEvent(streamB, ready, 0);     // B waits; the host does not
\`\`\`

The host keeps enqueuing. Only stream B's later work is held. \`cudaEventSynchronize\` is the version that does stop the host, used for timing.

## How many streams

Three stages (copy in, kernel, copy out) need **three streams** to keep all three units busy when the stages are similar lengths. A fourth stream has no fourth unit to fill, so the timeline stops getting shorter. When one stage dominates — a huge copy, or a kernel much longer than the copies — two streams already hit the bottleneck, and more streams copy the same engine's queue.

The lab is that timeline. \`pipeline.cu\` is the same loop on a real device.
`,b=`## Serial baseline

One stream, $N$ chunks, stages of length $H$ (host to device), $K$ (kernel) and $D$ (device to host):

$$
T_1 = N\\,(H + K + D).
$$

Nothing overlaps, pinned or not. The same total is what you get from pageable \`cudaMemcpyAsync\`: the host cannot stay ahead of the copies, so the streams never have two operations ready at once.

## Two copy engines, enough streams

Call the bottleneck $B = \\max(H, K, D)$. With pinned memory, separate H2D and D2H engines, and enough streams to keep every unit fed, the timeline is one pass of the short stages plus $N$ passes of the long one:

$$
T = N B + (H + K + D - B).
$$

The added term is the fill and the drain: the stages that are not the bottleneck run once beside the ends, instead of once per chunk.

For three stages of similar length, "enough" is **three streams**. Two streams cannot hold a copy-in, a kernel and a copy-out at the same time, so they land above this line. A fourth stream has nothing left to overlap and lands on it.

### Worked timelines

Stages are in the same time unit. Two engines unless noted. Pinned unless noted.

| $N$ | Streams | $H, K, D$ | Engines | $T$ | $T_1$ |
|---|---|---|---|---|---|
| 8 | 1 | 2, 2, 2 | 2 | 48 | 48 |
| 8 | 2 | 2, 2, 2 | 2 | 26 | 48 |
| 8 | 3 | 2, 2, 2 | 2 | 20 | 48 |
| 8 | 4 | 2, 2, 2 | 2 | 20 | 48 |
| 8 | 3 | 2, 2, 2 | 1 | 32 | 48 |
| 6 | 2 | 4, 2, 1 | 2 | 27 | 42 |
| 8 | 3 | 2, 2, 2 | 2, pageable | 48 | 48 |

Check the formula on the equal row: $B = 2$, $T = 8 \\cdot 2 + (6 - 2) = 20$. Three streams and four streams both hit it. Two streams finish at 26, because one of the three units is idle whenever the other two are busy.

The copy-bound row: $B = H = 4$, $T = 6 \\cdot 4 + (4 + 2 + 1 - 4) = 27$. The last chunk's kernel and copy-out are the only extra. A fourth stream still finishes at 27. The copy engine is already full.

One copy engine, equal stages: H2D and D2H never run together, so the copies alone cost $N(H + D) = 32$. Each kernel is the same length as one copy and hides beside the other direction. $T = 32$, not 20. Half the overlap disappeared with the second engine.

Pageable memory: $T = T_1$ at any stream count.

## Speedup

$$
\\frac{T_1}{T} = \\frac{H + K + D}{B + (H + K + D - B)/N}
\\;\\xrightarrow{N \\to \\infty}\\;
\\frac{H + K + D}{B}.
$$

Equal stages, two engines: the ceiling is $3\\times$, and $N = 8$ is already at $48/20 = 2.4\\times$. You do not get the ceiling at small $N$, because the fill and drain are still a visible fraction. One engine caps the same problem at $(H+K+D)/(H+D) = 1.5\\times$, which is exactly $48/32$.

## A dependency without a host stall

\`cudaStreamWaitEvent\` orders one operation in stream B behind one event in stream A. The host thread is not in that wait, so the enqueue of the rest of the pipeline continues. The GPU timeline gains a single edge. \`cudaDeviceSynchronize\` is the opposite edge: every stream, and the host, stop together.
`,x=`## Exercises

**Q1.** One stream, 8 chunks, each stage (H2D, kernel, D2H) takes 2 units. How long is the run?

<details>
<summary>Show answer</summary>

Nothing overlaps. $T = 8 \\times (2+2+2) = 48$.

</details>

**Q2.** Same chunks, pinned memory, two copy engines, three streams. How long, and what is the speedup against Q1?

<details>
<summary>Show answer</summary>

$B = 2$, so $T = 8 \\cdot 2 + (6 - 2) = 20$. Speedup $48/20 = 2.4\\times$. At that point a copy-in, a kernel and a copy-out are all busy.

</details>

**Q3.** Why does a fourth stream not beat those 20 units?

<details>
<summary>Show answer</summary>

There are three pieces of hardware and all three are already busy. The fourth stream only adds another queue for the same engines. The run stays at **20**.

</details>

**Q4.** Two streams instead of three, same equal stages. Why is the time 26 rather than 20?

<details>
<summary>Show answer</summary>

Two streams can keep only two of the three units busy. Whenever the kernel and one copy are running, the other copy has no chunk that is ready for it. You leave 6 units on the table: **26** instead of 20.

</details>

**Q5.** Six chunks, $H = 4$, $K = 2$, $D = 1$, two engines, two streams. Time? Would four streams do better?

<details>
<summary>Show answer</summary>

The copy-in is the bottleneck. $T = 6 \\cdot 4 + (2 + 1) = 27$. Four streams also finish at 27. The H2D engine is busy for the whole 24 units of copying; the last kernel and copy-out add 3, and another stream cannot split a single engine.

</details>

**Q6.** Eight equal chunks of length 2, three streams, but the host buffer came from \`malloc\`, not \`cudaMallocHost\`. Time?

<details>
<summary>Show answer</summary>

**48.** A pageable \`cudaMemcpyAsync\` can block the host, so the next operation is not queued in time to overlap. Stream count does not matter until the memory is pinned.

</details>

**Q7.** Same eight chunks, pinned, three streams, but the GPU has one copy engine. Time, and which overlap did you lose?

<details>
<summary>Show answer</summary>

H2D and D2H take turns. The copies cost $8 \\times (2+2) = 32$, and each kernel hides beside a copy, so **32**. You lost the overlap of the two directions. Speedup against 48 is $1.5\\times$, not $2.4\\times$.

</details>

**Q8.** Stream A runs the kernel. Stream B should copy the result back, and the host should keep enqueueing the next chunks. Which call orders B behind A without stopping the host?

<details>
<summary>Show answer</summary>

\`cudaEventRecord\` on A, then \`cudaStreamWaitEvent\` on B. \`cudaDeviceSynchronize\` would order them too, and it would also drain every stream and stall the host, which empties the pipeline.

</details>
`,S=`// nvcc -O3 -arch=sm_80 pipeline.cu -o pipeline
//
// Chunked saxpy. Each chunk is H2D, kernel, D2H on one stream.
// Chunks are dealt round-robin, so three streams keep the copy-in,
// the kernel and the copy-out in flight together.
// Reusing one device buffer per stream is safe: the next H2D on that
// stream is queued behind the previous D2H.

#include <cstdio>
#include <cuda_runtime.h>

__global__ void saxpy(float* y, const float* x, float a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a * x[i] + y[i];
}

#define CHECK(cmd) do { \\
    cudaError_t e = (cmd); \\
    if (e != cudaSuccess) { \\
        fprintf(stderr, "%s:%d %s\\n", __FILE__, __LINE__, cudaGetErrorString(e)); \\
        return 1; \\
    } \\
} while (0)

static float run(int nstreams, float* h_x, float* h_y, int n, int chunk, float a) {
    cudaError_t e;
    #define FAIL(cmd) do { e = (cmd); if (e != cudaSuccess) { \\
        fprintf(stderr, "%s:%d %s\\n", __FILE__, __LINE__, cudaGetErrorString(e)); \\
        return -1.f; } } while (0)
    cudaStream_t streams[4];
    float* d_x[4];
    float* d_y[4];
    for (int s = 0; s < nstreams; ++s) {
        FAIL(cudaStreamCreate(&streams[s]));
        FAIL(cudaMalloc(&d_x[s], chunk * sizeof(float)));
        FAIL(cudaMalloc(&d_y[s], chunk * sizeof(float)));
    }
    int block = 256;
    int grid = (chunk + block - 1) / block;
    cudaEvent_t start, stop;
    FAIL(cudaEventCreate(&start));
    FAIL(cudaEventCreate(&stop));
    FAIL(cudaEventRecord(start));
    int chunks = n / chunk;
    for (int c = 0; c < chunks; ++c) {
        int s = c % nstreams;
        size_t off = (size_t)c * chunk;
        FAIL(cudaMemcpyAsync(d_x[s], h_x + off, chunk * sizeof(float),
                              cudaMemcpyHostToDevice, streams[s]));
        FAIL(cudaMemcpyAsync(d_y[s], h_y + off, chunk * sizeof(float),
                              cudaMemcpyHostToDevice, streams[s]));
        saxpy<<<grid, block, 0, streams[s]>>>(d_y[s], d_x[s], a, chunk);
        FAIL(cudaMemcpyAsync(h_y + off, d_y[s], chunk * sizeof(float),
                              cudaMemcpyDeviceToHost, streams[s]));
    }
    FAIL(cudaEventRecord(stop));
    FAIL(cudaEventSynchronize(stop));   // one wait, after the whole pipeline is queued
    float ms = 0;
    FAIL(cudaEventElapsedTime(&ms, start, stop));
    for (int s = 0; s < nstreams; ++s) {
        cudaStreamDestroy(streams[s]);
        cudaFree(d_x[s]);
        cudaFree(d_y[s]);
    }
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms;
}

int main() {
    const int chunk = 1 << 20;
    const int chunks = 12;
    const int n = chunk * chunks;
    float *h_x, *h_y;
    CHECK(cudaMallocHost(&h_x, n * sizeof(float)));
    CHECK(cudaMallocHost(&h_y, n * sizeof(float)));
    for (int i = 0; i < n; ++i) { h_x[i] = 1.f; h_y[i] = 2.f; }

    float one = run(1, h_x, h_y, n, chunk, 3.f);
    for (int i = 0; i < n; ++i) h_y[i] = 2.f;
    float three = run(3, h_x, h_y, n, chunk, 3.f);
    printf("1 stream  %.3f ms\\n", one);
    printf("3 streams %.3f ms\\n", three);
    printf("speedup   %.2fx   (copies are small next to a tiny saxpy — raise the chunk or the\\n"
           "                    arithmetic if the kernel finishes before the next copy is queued)\\n",
           one / three);

    cudaFreeHost(h_x);
    cudaFreeHost(h_y);
    return 0;
}
`,C=`// nvcc -O3 -arch=sm_80 events.cu -o events
//
// Stream B's copy waits for stream A's kernel. The host does not.
// cudaDeviceSynchronize() would also order them, and it would drain
// every other stream at the same time.

#include <cstdio>
#include <cuda_runtime.h>

__global__ void fill(float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = 1.f;
}

#define CHECK(cmd) do { \\
    cudaError_t e = (cmd); \\
    if (e != cudaSuccess) { \\
        fprintf(stderr, "%s:%d %s\\n", __FILE__, __LINE__, cudaGetErrorString(e)); \\
        return 1; \\
    } \\
} while (0)

int main() {
    const int n = 1 << 20;
    float *d, *h;
    cudaStream_t a, b;
    cudaEvent_t ready;
    CHECK(cudaMalloc(&d, n * sizeof(float)));
    CHECK(cudaMallocHost(&h, n * sizeof(float)));
    CHECK(cudaStreamCreateWithFlags(&a, cudaStreamNonBlocking));
    CHECK(cudaStreamCreateWithFlags(&b, cudaStreamNonBlocking));
    CHECK(cudaEventCreateWithFlags(&ready, cudaEventDisableTiming));

    fill<<<(n + 255) / 256, 256, 0, a>>>(d, n);
    CHECK(cudaEventRecord(ready, a));          // becomes signaled when \`fill\` finishes
    CHECK(cudaStreamWaitEvent(b, ready, 0));   // b's later work waits; the host returns now
    CHECK(cudaMemcpyAsync(h, d, n * sizeof(float), cudaMemcpyDeviceToHost, b));

    // The host could enqueue more chunks here. Nothing above has stopped it.
    CHECK(cudaStreamSynchronize(b));
    printf("h[0] = %.0f (1 means B's copy saw A's kernel)\\n", h[0]);

    cudaEventDestroy(ready);
    cudaStreamDestroy(a);
    cudaStreamDestroy(b);
    cudaFreeHost(h);
    cudaFree(d);
    return 0;
}
`,w=`// nvcc -O3 -arch=sm_80 pinned.cu -o pinned
//
// Pageable cudaMemcpyAsync is allowed to block the host: the driver
// copies into an internal pinned bounce buffer first. A buffer from
// cudaMallocHost is already DMA-able, so the call returns and the copy
// engine overlaps whatever kernel is running in another stream.
//
// Time the two against the pipeline in pipeline.cu. This file only
// shows the allocation, and the flag that keeps a stream out of the
// legacy default-stream barrier.

#include <cstdio>
#include <cuda_runtime.h>

#define CHECK(cmd) do { \\
    cudaError_t e = (cmd); \\
    if (e != cudaSuccess) { \\
        fprintf(stderr, "%s:%d %s\\n", __FILE__, __LINE__, cudaGetErrorString(e)); \\
        return 1; \\
    } \\
} while (0)

int main() {
    const size_t n = 1 << 20;
    float* pageable = new float[n];
    float* pinned = nullptr;
    float* d = nullptr;
    CHECK(cudaMallocHost(&pinned, n * sizeof(float)));   // pair with cudaFreeHost
    CHECK(cudaMalloc(&d, n * sizeof(float)));

    cudaStream_t stream;
    // Non-blocking: this stream does not wait for, and is not waited on by,
    // work launched into the legacy default stream (the NULL stream).
    CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    CHECK(cudaMemcpyAsync(d, pinned, n * sizeof(float), cudaMemcpyHostToDevice, stream));
    // cudaMemcpyAsync(d, pageable, ...) can return only after a staging copy.
    // Don't use it on the overlap path.

    CHECK(cudaStreamSynchronize(stream));
    printf("pinned %zu MB, stream is non-blocking\\n", (n * sizeof(float)) >> 20);
    printf("pageable pointer %p was not used for the DMA\\n", (void*)pageable);

    cudaStreamDestroy(stream);
    cudaFree(d);
    cudaFreeHost(pinned);
    delete[] pageable;
    return 0;
}
`,T=n(),E={h:`H2D`,k:`kernel`,d:`D2H`},D={h:i.b,k:i.e,d:i.d},O=[`#4ade80`,`#22d3ee`,`#fbbf24`,`#f472b6`,`#a78bfa`,`#fb7185`,`#86efac`,`#67e8f9`];function k(e){let[t,n]=(0,g.useState)(e.n),[r,i]=(0,g.useState)(e.streams),[a,o]=(0,g.useState)(e.h),[s,c]=(0,g.useState)(e.k),[l,u]=(0,g.useState)(e.d),[d,f]=(0,g.useState)(!0),[p,m]=(0,g.useState)(!0);return{n:t,setN:n,streams:r,setStreams:i,h:a,setH:o,k:s,setK:c,d:l,setD:u,pinned:d,setPinned:f,split:p,setSplit:m,sched:(0,g.useMemo)(()=>_({n:t,streams:r,h:a,k:s,d:l,pinned:d,splitEngines:p}),[t,r,a,s,l,d,p])}}function A({p:e,extra:t}){return(0,T.jsxs)(l,{children:[(0,T.jsx)(c,{label:`Chunks`,min:1,max:8,value:e.n,onChange:e.setN}),(0,T.jsx)(c,{label:`Streams`,min:1,max:4,value:e.streams,onChange:e.setStreams}),(0,T.jsx)(c,{label:`H2D`,min:1,max:8,value:e.h,onChange:e.setH}),(0,T.jsx)(c,{label:`Kernel`,min:1,max:8,value:e.k,onChange:e.setK}),(0,T.jsx)(c,{label:`D2H`,min:1,max:8,value:e.d,onChange:e.setD}),(0,T.jsx)(h,{label:`Pinned memory`,value:e.pinned,onChange:e.setPinned}),(0,T.jsx)(h,{label:`Two copy engines`,value:e.split,onChange:e.setSplit}),t]})}function j(){let e=k({n:8,streams:1,h:2,k:2,d:2}),{sched:t}=e,n=t.serialTotal/t.total,[o]=m(360,(e,o)=>{let s=o-62-14,c=[`h`,`k`,`d`];a(e,t.total+` units`,62,16,{size:13,color:n>1.05?i.e:i.d,weight:700}),a(e,`serial `+t.serialTotal,o-14,16,{size:12,color:i.mute,align:`right`,mono:!0});let l=s/t.total;c.forEach((n,i)=>{let o=36+i*78;a(e,E[n],10,o+22,{size:12,color:D[n],weight:700}),e.fillStyle=`#14182a`,r(e,62,o,s,44,6),e.fill(),t.ops.filter(e=>e.kind===n).forEach(t=>{let n=62+t.start*l,i=Math.max(2,(t.end-t.start)*l-1);e.fillStyle=O[t.i%O.length],r(e,n,o+6,i,32,4),e.fill(),i>16&&a(e,String(t.i),n+i/2,o+22,{size:12,color:`#0a0c14`,align:`center`,weight:800,mono:!0})})}),a(e,`each bar is one chunk  ·  a gap on a row is that unit sitting idle`,62,280,{size:11,color:i.mute})});return(0,T.jsxs)(T.Fragment,{children:[(0,T.jsx)(`canvas`,{...o}),(0,T.jsx)(A,{p:e}),(0,T.jsxs)(`div`,{className:`controls`,children:[(0,T.jsx)(f,{onClick:()=>{e.setN(8),e.setStreams(1),e.setH(2),e.setK(2),e.setD(2)},children:`1 stream`}),(0,T.jsx)(f,{onClick:()=>{e.setN(8),e.setStreams(3),e.setH(2),e.setK(2),e.setD(2)},children:`3 streams, equal`}),(0,T.jsx)(f,{onClick:()=>{e.setN(6),e.setStreams(2),e.setH(4),e.setK(2),e.setD(1)},children:`copy bound`})]}),(0,T.jsx)(s,{items:[[i.b,`H2D row`],[i.e,`kernel row`],[i.d,`D2H row`]]}),(0,T.jsxs)(d,{children:[`Pipeline `,(0,T.jsx)(`b`,{children:t.total}),` · one stream `,(0,T.jsx)(`b`,{children:t.serialTotal}),` · speedup `,(0,T.jsxs)(`b`,{className:n>1.05?`g`:`w`,children:[n.toFixed(2),`×`]}),`.`,!e.pinned&&(0,T.jsx)(T.Fragment,{children:` Pinned is off, so every copy blocks the host and the streams never get ahead. The run is serial.`}),e.pinned&&!e.split&&(0,T.jsx)(T.Fragment,{children:` One copy engine: H2D and D2H take turns. Only the kernel can overlap a copy.`}),e.pinned&&e.split&&e.h===e.k&&e.k===e.d&&e.streams===2&&e.n>2&&(0,T.jsx)(T.Fragment,{children:` Two streams can only fill two of the three units, so this run sits above the three-stream line.`}),e.pinned&&e.split&&e.streams>=3&&e.h===e.k&&e.k===e.d&&(0,T.jsxs)(T.Fragment,{children:[` Three equal stages and three busy units. T = N·B + (H+K+D − B) = `,t.total,`.`]}),e.pinned&&e.split&&e.streams>=2&&e.h>e.k&&e.h>e.d&&(0,T.jsx)(T.Fragment,{children:` H2D is the bottleneck. Streams past two do not shorten a copy engine that is already full.`})]})]})}function M({ops:e,t}){let n=v(e,t),r={h:-3,k:0,d:3};return(0,T.jsxs)(`group`,{children:[[`h`,`k`,`d`].map(e=>(0,T.jsxs)(`mesh`,{position:[r[e],-.05,0],children:[(0,T.jsx)(`boxGeometry`,{args:[2.2,.12,1.6]}),(0,T.jsx)(`meshStandardMaterial`,{color:`#1a2033`,roughness:.6})]},e)),n.map(e=>(0,T.jsxs)(`mesh`,{position:[r[e.kind],.55,0],children:[(0,T.jsx)(`boxGeometry`,{args:[1.15,.9,.9]}),(0,T.jsx)(`meshStandardMaterial`,{color:O[e.i%O.length],emissive:O[e.i%O.length],emissiveIntensity:.35,roughness:.4})]},e.kind+e.i))]})}function N(){let e=k({n:8,streams:3,h:2,k:2,d:2}),[t,n]=(0,g.useState)(4),r=Math.min(t,e.sched.total),a=[`h`,`k`,`d`].map(t=>v(e.sched.ops,r).find(e=>e.kind===t)).filter(Boolean),s=a.length?a.map(e=>E[e.kind]+` chunk `+e.i).join(`   `):`pipeline empty`,l=(0,T.jsxs)(T.Fragment,{children:[`t = `,(0,T.jsx)(`b`,{style:{color:i.b},children:r}),` / `,e.sched.total,(0,T.jsx)(`br`,{}),s]});return(0,T.jsxs)(T.Fragment,{children:[(0,T.jsx)(o,{height:420,camera:[0,3.6,8.2],target:[0,.3,0],fov:42,overlay:l,hint:`drag to orbit · left platform is H2D, middle is the kernel, right is D2H`,children:(0,T.jsx)(M,{ops:e.sched.ops,t:r})}),(0,T.jsx)(A,{p:e,extra:(0,T.jsx)(c,{label:`Time`,min:0,max:Math.max(1,e.sched.total),value:r,onChange:n})}),(0,T.jsx)(d,{children:e.streams===1?(0,T.jsx)(T.Fragment,{children:`One stream: only one platform is ever occupied. Step time and the chunk walks left to right.`}):(0,T.jsxs)(T.Fragment,{children:[`With `,e.streams,` streams, two or three platforms can hold a chunk at the same moment. That is the overlap.`]})})]})}var P=[{id:`same`,label:`two kernels, one stream`,lanes:[`stream 1`],overlap:!1,ops:[{lane:0,kind:`k`,start:0,end:4,tag:`A`},{lane:0,kind:`k`,start:4,end:8,tag:`B`}],why:`A stream is ordered. B does not reach the SMs until A has finished.`},{id:`pin`,label:`pinned H2D beside a kernel`,lanes:[`stream 1`,`stream 2`],overlap:!0,ops:[{lane:0,kind:`k`,start:0,end:5,tag:`kernel`},{lane:1,kind:`h`,start:0,end:5,tag:`H2D`}],why:`Different streams, pinned host memory, copy engine free. The DMA and the SMs run together.`},{id:`page`,label:`pageable H2D beside a kernel`,lanes:[`stream 1`,`stream 2`],overlap:!1,ops:[{lane:0,kind:`k`,start:0,end:4,tag:`kernel`},{lane:1,kind:`h`,start:4,end:8,tag:`H2D`}],why:`cudaMemcpyAsync from pageable memory can block the host while the driver stages the bytes. The kernel is finished before the copy starts.`},{id:`def`,label:`legacy default stream`,lanes:[`stream 1`,`default`],overlap:!1,ops:[{lane:0,kind:`k`,start:0,end:4,tag:`kernel`},{lane:1,kind:`h`,start:4,end:8,tag:`H2D`}],why:`Work in the legacy default stream waits for every blocking stream, then blocks them. Launching the copy with no stream argument puts it here.`},{id:`evt`,label:`wait on one event`,lanes:[`stream A`,`stream B`],overlap:!1,ops:[{lane:0,kind:`k`,start:0,end:4,tag:`kernel`},{lane:1,kind:`d`,start:4,end:7,tag:`D2H`}],why:`cudaStreamWaitEvent holds B’s copy until A’s kernel finishes. The host is not held: it can keep enqueueing. cudaDeviceSynchronize would have stopped the host and every stream.`}];function F(){let[e,t]=(0,g.useState)(`pin`),n=P.find(t=>t.id===e),o=Math.max(...n.ops.map(e=>e.end)),[s]=m(260,(e,t)=>{let s=t-78-16,c=s/o;a(e,n.overlap?`overlaps`:`does not overlap`,78,16,{size:13,color:n.overlap?i.e:i.d,weight:700}),n.lanes.forEach((t,o)=>{let l=40+o*78;a(e,t,8,l+20,{size:11,color:i.mute}),e.fillStyle=`#14182a`,r(e,78,l,s,40,6),e.fill(),n.ops.filter(e=>e.lane===o).forEach(t=>{let n=78+t.start*c,i=Math.max(8,(t.end-t.start)*c-2);e.fillStyle=D[t.kind],r(e,n,l+6,i,28,4),e.fill(),a(e,t.tag,n+i/2,l+20,{size:12,color:`#0a0c14`,align:`center`,weight:700})})})});return(0,T.jsxs)(T.Fragment,{children:[(0,T.jsx)(`canvas`,{...s}),(0,T.jsx)(l,{children:(0,T.jsx)(p,{label:`Launch`,value:e,onChange:t,options:P.map(e=>[e.id,e.label])})}),(0,T.jsx)(d,{children:n.why})]})}function I(){return(0,T.jsx)(u,{views:[{id:`t`,label:`Timeline`,render:()=>(0,T.jsx)(j,{})},{id:`d`,label:`3D: three platforms`,render:()=>(0,T.jsx)(N,{})},{id:`r`,label:`What may overlap`,render:()=>(0,T.jsx)(F,{})}]})}var L={Lab:I,vizTitle:`Queue chunks across streams and watch the copy engines overlap the kernel`,tryIt:[`Leave **1 stream**, 8 chunks, every stage 2. The three rows never run together. The run is 48.`,`Click **3 streams, equal**. A copy-in, a kernel and a copy-out are busy at once. The run drops to 20, which is 2.40×.`,`Drag **Streams** to 2, then to 4. Two streams finish at 26 — one unit is always idle. Four streams stay at 20. There is no fourth unit to fill.`,`Turn **Two copy engines** off. H2D and D2H stop sharing the timeline and the run goes to 32.`,`Turn **Pinned memory** off. Every stream count collapses back to 48. The host cannot enqueue ahead of a pageable copy.`,`Open **3D** (it starts at 3 streams). Step **Time**. Two or three platforms hold a chunk at once. Switch to 1 stream and only one platform is ever lit.`,`Open **What may overlap**. Pinned H2D beside a kernel shares the clock. The legacy default stream and pageable memory do not.`],theory:y,math:b,practice:x,code:[{title:`Chunked saxpy on 1 stream and on 3`,lang:`cuda`,note:`Pinned buffers, one device buffer per stream, every memcpy and the kernel launched into that stream. One event wait after the whole loop.`,src:S},{title:`Order stream B behind stream A without stopping the host`,lang:`cuda`,note:`cudaEventRecord on A, cudaStreamWaitEvent on B. The host returns immediately and can enqueue the next chunks.`,src:C},{title:`Pinned allocation, and a stream the default stream cannot see`,lang:`cuda`,note:`cudaMallocHost is the overlap path. cudaStreamNonBlocking does not join the legacy default-stream barrier.`,src:w}],quiz:[{q:`Inside one stream, a kernel and the copy that follows it:`,options:[`Run at the same time`,`Run in order`,`Run only if the memory is pinned`,`Require an event`],answer:1,why:`A stream is a queue. Overlap happens across streams, not inside one.`},{q:`cudaMemcpyAsync from memory returned by malloc:`,options:[`Always overlaps the previous kernel`,`Can block the host while the driver stages the bytes`,`Is a compile error`,`Uses both copy engines`],answer:1,why:`Pageable memory is not DMA-able. The driver may copy it into a pinned bounce buffer before the call returns, so the kernel you meant to overlap is already done.`},{q:`Eight chunks, each stage length 2, three streams, two copy engines, pinned. The run takes:`,options:[`48`,`26`,`20`,`16`],answer:2,why:`$T = NB + (H+K+D-B) = 16 + 4 = 20$. One stream would take 48.`},{q:`Why does a fourth stream not improve that 20?`,options:[`Streams are capped at 3`,`All three units are already busy`,`The fourth stream deadlocks`,`Pinned memory allows only three copies`],answer:1,why:`The H2D engine, the SMs and the D2H engine are the whole machine. A fourth queue has nothing new to occupy.`},{q:`A kernel launched with no stream argument, beside an H2D in a blocking stream:`,options:[`Overlaps the copy`,`Goes to the legacy default stream, which serializes with blocking streams`,`Is pinned automatically`,`Runs on the copy engine`],answer:1,why:`The omitted stream is the legacy default stream. It waits for blocking streams and then blocks them.`},{q:`cudaStreamWaitEvent, compared with cudaDeviceSynchronize:`,options:[`Stops the host until every stream finishes`,`Orders one stream behind one event and lets the host keep enqueueing`,`Pins the buffer`,`Starts a copy engine`],answer:1,why:`The wait is on the GPU queue. The host thread is not in it, so the rest of the pipeline can still be queued.`},{q:`Two fat kernels, each of which already fills the SMs, in two streams. They:`,options:[`Always overlap`,`Overlap only if both fit at once, which they do not`,`Require pinned memory to overlap`,`Share a warp`],answer:1,why:`Different streams are necessary and not sufficient. Concurrent kernels need leftover SMs. Copy/compute overlap is the one that does not.`}]};export{L as default};