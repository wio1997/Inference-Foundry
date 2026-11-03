#include "host_packet_snapshot.h"
#include <assert.h>
#include <stdio.h>
#include <string.h>

int main(void) {
    unsigned char data[4097];for(size_t i=0;i<sizeof(data);i++)data[i]=(unsigned char)(i*7);
    wio_host_snapshot out,before;wio_placeholder p={8,24};size_t offset=999;
    memset(&out,0xa5,sizeof(out));before=out;
    assert(wio_snapshot_host_args(0,WIO_FROM_LIVE_HOST_ARGS_API,(void*)1,152,(void*)1,1,&out)==WIO_DISABLED);
    assert(memcmp(&out,&before,sizeof(out))==0);
    assert(wio_snapshot_host_args(1,WIO_FROM_DEVICE_ARGS_OR_GETTER,(void*)1,152,(void*)1,1,&out)==WIO_NOT_HOST_CALL);
    assert(memcmp(&out,&before,sizeof(out))==0);
    size_t sizes[]={0,1,8,24,96,152,320,4096};
    for(size_t i=0;i<sizeof(sizes)/sizeof(sizes[0]);i++) {
        size_t n=sizes[i];
        assert(wio_snapshot_host_args(1,WIO_FROM_LIVE_HOST_ARGS_API,data,n,NULL,0,&out)==WIO_COPY_OK);
        assert(out.args_size==n && out.placeholder_count==0 && memcmp(out.args,data,n)==0);
        for(size_t j=n;j<WIO_MAX_HOST_ARGS;j++)assert(out.args[j]==0);
    }
    assert(wio_snapshot_host_args(1,WIO_FROM_LIVE_HOST_ARGS_API,data,152,&p,1,&out)==WIO_COPY_OK);
    assert(wio_inline_data(&out,8,96,&offset)==WIO_COPY_OK && offset==24);
    assert(wio_inline_data(&out,0,96,&offset)==WIO_BAD_PLACEHOLDER);
    assert(wio_inline_data(&out,8,129,&offset)==WIO_BAD_PLACEHOLDER);
    out.placeholders[1]=out.placeholders[0];out.placeholder_count=2;
    assert(wio_inline_data(&out,8,96,&offset)==WIO_BAD_PLACEHOLDER);
    out.placeholder_count=1;
    out.placeholders[0].addrOffset=145;
    assert(wio_inline_data(&out,145,96,&offset)==WIO_BAD_PLACEHOLDER);
    assert(wio_snapshot_host_args(1,WIO_FROM_LIVE_HOST_ARGS_API,data,4097,NULL,0,&out)==WIO_BUDGET_EXCEEDED);
    assert(wio_snapshot_host_args(1,WIO_FROM_LIVE_HOST_ARGS_API,NULL,1,NULL,0,&out)==WIO_BAD_ARGUMENT);
    assert(wio_snapshot_host_args(1,WIO_FROM_LIVE_HOST_ARGS_API,data,152,NULL,1,&out)==WIO_BAD_ARGUMENT);
    p.addrOffset=145;
    assert(wio_snapshot_host_args(1,WIO_FROM_LIVE_HOST_ARGS_API,data,152,&p,1,&out)==WIO_BAD_PLACEHOLDER);
    p.addrOffset=8;p.dataOffset=153;
    assert(wio_snapshot_host_args(1,WIO_FROM_LIVE_HOST_ARGS_API,data,152,&p,1,&out)==WIO_BAD_PLACEHOLDER);
    assert(wio_snapshot_host_args(1,WIO_FROM_LIVE_HOST_ARGS_API,&out,152,NULL,0,&out)==WIO_BAD_ARGUMENT);
    assert(wio_snapshot_host_args(1,WIO_FROM_LIVE_HOST_ARGS_API,data,152,out.placeholders,1,&out)==WIO_BAD_ARGUMENT);
    puts("{\"status\":\"PASS\",\"scope\":\"CPU_CORE_ONLY_NO_CANN_OR_NPU\",\"size_cases\":8,\"source_and_bounds_guards\":true}");
    return 0;
}
