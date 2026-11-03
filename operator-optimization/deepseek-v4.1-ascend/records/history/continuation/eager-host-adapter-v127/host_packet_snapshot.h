#ifndef WIO_HOST_PACKET_SNAPSHOT_H
#define WIO_HOST_PACKET_SNAPSHOT_H
#include <stddef.h>
#include <stdint.h>

#define WIO_MAX_HOST_ARGS 4096u
#define WIO_MAX_PLACEHOLDERS 32u
/* Mirrors only the pinned public two-uint32 placeholder record. */
typedef struct { uint32_t addrOffset; uint32_t dataOffset; } wio_placeholder;
typedef struct {
    size_t args_size;
    size_t placeholder_count;
    unsigned char args[WIO_MAX_HOST_ARGS];
    wio_placeholder placeholders[WIO_MAX_PLACEHOLDERS];
} wio_host_snapshot;
enum { WIO_COPY_OK=0, WIO_DISABLED=1, WIO_NOT_HOST_CALL=2,
       WIO_BAD_ARGUMENT=3, WIO_BUDGET_EXCEEDED=4, WIO_BAD_PLACEHOLDER=5 };
enum { WIO_FROM_LIVE_HOST_ARGS_API=1, WIO_FROM_DEVICE_ARGS_OR_GETTER=2 };

/* Pure CPU core,not an installed hook. Caller must supply valid live host spans
 * and a separate owned output object. Embedded address words are never read. */
int wio_snapshot_host_args(int enabled,int source_kind,const void *host_args,size_t args_size,
    const wio_placeholder *placeholders,size_t placeholder_count,wio_host_snapshot *output);
/* Resolve an explicitly supplied inline-data relocation by offsets only. */
int wio_inline_data(const wio_host_snapshot *snapshot,uint32_t addr_offset,size_t required_size,
    size_t *data_offset);
#endif
