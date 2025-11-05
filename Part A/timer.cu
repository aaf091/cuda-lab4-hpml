// timer.cu
#include <cuda_runtime.h>
#include <stdio.h>

static cudaEvent_t startEvent, stopEvent;

void start_timer() {
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);
    cudaEventRecord(startEvent, 0);
}
float stop_timer() {
    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);
    float ms;
    cudaEventElapsedTime(&ms, startEvent, stopEvent);
    return ms;
}