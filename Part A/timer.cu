#include <cuda_runtime.h>
#include "timer.h"

static cudaEvent_t evStart, evStop;
static bool inited = false;

void start_timer() {
    if (!inited) {
        cudaEventCreate(&evStart);
        cudaEventCreate(&evStop);
        inited = true;
    }
    cudaEventRecord(evStart, 0);
}

float stop_timer() {
    cudaEventRecord(evStop, 0);
    cudaEventSynchronize(evStop);
    float ms = 0.f;
    cudaEventElapsedTime(&ms, evStart, evStop);
    return ms;
}
