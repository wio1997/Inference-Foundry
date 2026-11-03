#include "host_packet_snapshot.h"
#include <limits.h>
#include <string.h>

_Static_assert(sizeof(void*)==8,"LP64 reference required");
_Static_assert(sizeof(size_t)==8,"LP64 size_t reference required");
_Static_assert(sizeof(wio_placeholder)==8,"placeholder ABI size");
_Static_assert(offsetof(wio_placeholder,addrOffset)==0,"placeholder addr offset");
_Static_assert(offsetof(wio_placeholder,dataOffset)==4,"placeholder data offset");

static int span_end(const void *p,size_t n,uintptr_t *end) {
    uintptr_t start=(uintptr_t)p;
    if (n>UINTPTR_MAX-start) return 0;
    *end=start+n;
    return 1;
}
static int overlaps(const void *a,size_t an,const void *b,size_t bn) {
    uintptr_t ae,be;
    if (!span_end(a,an,&ae)||!span_end(b,bn,&be)) return 1;
    return an && bn && ((uintptr_t)a<be) && ((uintptr_t)b<ae);
}
int wio_snapshot_host_args(int enabled,int source_kind,const void *host_args,size_t args_size,
    const wio_placeholder *placeholders,size_t placeholder_count,wio_host_snapshot *output) {
    if (!enabled) return WIO_DISABLED;
    if (source_kind!=WIO_FROM_LIVE_HOST_ARGS_API) return WIO_NOT_HOST_CALL;
    if (!output || (args_size&&!host_args) || (placeholder_count&&!placeholders)) return WIO_BAD_ARGUMENT;
    if (args_size>WIO_MAX_HOST_ARGS || placeholder_count>WIO_MAX_PLACEHOLDERS) return WIO_BUDGET_EXCEEDED;
    if (overlaps(host_args,args_size,output,sizeof(*output)) ||
        overlaps(placeholders,placeholder_count*sizeof(*placeholders),output,sizeof(*output))) return WIO_BAD_ARGUMENT;
    for (size_t i=0;i<placeholder_count;i++) {
        size_t a=placeholders[i].addrOffset,d=placeholders[i].dataOffset;
        if (args_size<sizeof(void*) || a>args_size-sizeof(void*) || d>args_size) return WIO_BAD_PLACEHOLDER;
    }
    memset(output,0,sizeof(*output));
    output->args_size=args_size;
    output->placeholder_count=placeholder_count;
    if (args_size) memcpy(output->args,host_args,args_size);
    if (placeholder_count) memcpy(output->placeholders,placeholders,placeholder_count*sizeof(*placeholders));
    return WIO_COPY_OK;
}
int wio_inline_data(const wio_host_snapshot *snapshot,uint32_t addr_offset,size_t required_size,
    size_t *data_offset) {
    if (!snapshot || !data_offset || snapshot->args_size>WIO_MAX_HOST_ARGS ||
        snapshot->placeholder_count>WIO_MAX_PLACEHOLDERS) return WIO_BAD_ARGUMENT;
    if (snapshot->args_size<sizeof(void*) || addr_offset>snapshot->args_size-sizeof(void*))
        return WIO_BAD_PLACEHOLDER;
    size_t found=0,offset=0;
    for (size_t i=0;i<snapshot->placeholder_count;i++) {
        if (snapshot->placeholders[i].addrOffset==addr_offset) {
            offset=snapshot->placeholders[i].dataOffset;found++;
        }
    }
    if (found!=1 || offset>snapshot->args_size || required_size>snapshot->args_size-offset)
        return WIO_BAD_PLACEHOLDER;
    *data_offset=offset;
    return WIO_COPY_OK;
}
