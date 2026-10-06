#include <stdint.h>
#include <stdatomic.h>
#include <time.h>
static uint64_t ns(clockid_t id) { struct timespec t; clock_gettime(id,&t); return (uint64_t)t.tv_sec*1000000000+t.tv_nsec; }
uint64_t perf_head_acquire(const uint64_t *p){return atomic_load_explicit((const _Atomic uint64_t*)p,memory_order_acquire);}
void perf_tail_release(uint64_t *p,uint64_t value){atomic_store_explicit((_Atomic uint64_t*)p,value,memory_order_release);}
__attribute__((noinline)) uint64_t fixture_workspace(void){uint64_t begin=ns(CLOCK_THREAD_CPUTIME_ID),v=0;do {for(int i=0;i<10000;i++)v+=i;}while(ns(CLOCK_THREAD_CPUTIME_ID)-begin<1000000); return v;}
__attribute__((noinline)) uint64_t fixture_predicate(void){struct timespec t={0,2000000};nanosleep(&t,0);return 7;}
__attribute__((noinline)) uint64_t fixture_outer(uint64_t *out){out[0]=ns(CLOCK_MONOTONIC);out[1]=ns(CLOCK_THREAD_CPUTIME_ID);uint64_t v=fixture_predicate()+fixture_workspace();out[2]=ns(CLOCK_THREAD_CPUTIME_ID);out[3]=ns(CLOCK_MONOTONIC);return v;}
