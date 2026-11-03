#include <acl/acl_rt.h>
#include <pthread.h>
#include <stdlib.h>
static pthread_barrier_t gate;
static void *worker(void *x){(void)x;aclrtBinHandle b;aclrtFuncHandle f;pthread_barrier_wait(&gate);if(aclrtBinaryLoadFromData("abcdefghij",10,NULL,&b)||aclrtBinaryGetFunction(b,"DispatchFFNCombine_actual",&f)||aclrtBinaryUnLoad(b))abort();return NULL;}
int main(void){pthread_t a,b;pthread_barrier_init(&gate,NULL,2);pthread_create(&a,NULL,worker,NULL);pthread_create(&b,NULL,worker,NULL);pthread_join(a,NULL);pthread_join(b,NULL);pthread_barrier_destroy(&gate);return 0;}
