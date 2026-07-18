#include <stdint.h>

/*
 * Minimal ITT/JIT profiling shim for environments where PyTorch is linked
 * against the JIT profiling API but the runtime library is absent.
 *
 * These no-op implementations are sufficient to let torch import and run.
 */

typedef enum {
    iJVM_EVENT_TYPE_SHUTDOWN = 2
} iJIT_JVM_EVENT;

int iJIT_NotifyEvent(iJIT_JVM_EVENT event_type, void *event_specific_data) {
    (void)event_type;
    (void)event_specific_data;
    return 0;
}

int iJIT_IsProfilingActive(void) {
    return 0;
}

unsigned int iJIT_GetNewMethodID(void) {
    static unsigned int next_id = 1;
    return next_id++;
}
