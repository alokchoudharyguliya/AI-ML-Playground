## Exercises

**Q1.** One stream, 8 chunks, each stage (H2D, kernel, D2H) takes 2 units. How long is the run?

<details>
<summary>Show answer</summary>

Nothing overlaps. $T = 8 \times (2+2+2) = 48$.

</details>

**Q2.** Same chunks, pinned memory, two copy engines, three streams. How long, and what is the speedup against Q1?

<details>
<summary>Show answer</summary>

$B = 2$, so $T = 8 \cdot 2 + (6 - 2) = 20$. Speedup $48/20 = 2.4\times$. At that point a copy-in, a kernel and a copy-out are all busy.

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

The copy-in is the bottleneck. $T = 6 \cdot 4 + (2 + 1) = 27$. Four streams also finish at 27. The H2D engine is busy for the whole 24 units of copying; the last kernel and copy-out add 3, and another stream cannot split a single engine.

</details>

**Q6.** Eight equal chunks of length 2, three streams, but the host buffer came from `malloc`, not `cudaMallocHost`. Time?

<details>
<summary>Show answer</summary>

**48.** A pageable `cudaMemcpyAsync` can block the host, so the next operation is not queued in time to overlap. Stream count does not matter until the memory is pinned.

</details>

**Q7.** Same eight chunks, pinned, three streams, but the GPU has one copy engine. Time, and which overlap did you lose?

<details>
<summary>Show answer</summary>

H2D and D2H take turns. The copies cost $8 \times (2+2) = 32$, and each kernel hides beside a copy, so **32**. You lost the overlap of the two directions. Speedup against 48 is $1.5\times$, not $2.4\times$.

</details>

**Q8.** Stream A runs the kernel. Stream B should copy the result back, and the host should keep enqueueing the next chunks. Which call orders B behind A without stopping the host?

<details>
<summary>Show answer</summary>

`cudaEventRecord` on A, then `cudaStreamWaitEvent` on B. `cudaDeviceSynchronize` would order them too, and it would also drain every stream and stall the host, which empties the pipeline.

</details>
