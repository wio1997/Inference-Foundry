#define _GNU_SOURCE
/* Process-local successful file-open observation; no payload or unrelated path logging. */
#include <dlfcn.h>
#include <fcntl.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/syscall.h>
#include <unistd.h>
static __thread int tracing;
static void note(const char *p) {
    if (tracing || !p || !(strstr(p,"dequant_swiglu_quant") || strstr(p,"DequantSwigluQuant") || strstr(p,"libcust_opapi.so"))) return;
    tracing=1;
    const char *dest=getenv("WIO_KERNEL_OPEN_LOG");
    if (dest) {
        int fd=syscall(SYS_openat,AT_FDCWD,dest,O_WRONLY|O_CREAT|O_APPEND,0600);
        if(fd>=0){syscall(SYS_write,fd,p,strlen(p));syscall(SYS_write,fd,"\n",1);syscall(SYS_close,fd);}
    }
    tracing=0;
}
#define DEFINE_OPEN(name) \
int name(const char *p,int flags,...){ \
 static int (*real)(const char*,int,...);if(!real)real=dlsym(RTLD_NEXT,#name); \
 mode_t mode=0;if((flags&O_CREAT)||((flags&O_TMPFILE)==O_TMPFILE)){va_list ap;va_start(ap,flags);mode=va_arg(ap,int);va_end(ap);} \
 int result=real(p,flags,mode);if(result>=0)note(p);return result;}
DEFINE_OPEN(open)
DEFINE_OPEN(open64)
#define DEFINE_OPENAT(name) \
int name(int dir,const char *p,int flags,...){ \
 static int (*real)(int,const char*,int,...);if(!real)real=dlsym(RTLD_NEXT,#name); \
 mode_t mode=0;if((flags&O_CREAT)||((flags&O_TMPFILE)==O_TMPFILE)){va_list ap;va_start(ap,flags);mode=va_arg(ap,int);va_end(ap);} \
 int result=real(dir,p,flags,mode);if(result>=0)note(p);return result;}
DEFINE_OPENAT(openat)
DEFINE_OPENAT(openat64)
#define DEFINE_FOPEN(name) \
FILE *name(const char *p,const char *mode){static FILE *(*real)(const char*,const char*);if(!real)real=dlsym(RTLD_NEXT,#name);FILE *r=real(p,mode);if(r)note(p);return r;}
DEFINE_FOPEN(fopen)
DEFINE_FOPEN(fopen64)
