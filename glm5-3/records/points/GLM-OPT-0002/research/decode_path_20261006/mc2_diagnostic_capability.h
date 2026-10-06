// Diagnostic A/B control around the same fixed-lifetime capability predicate.
// Production diff is mc2_capability.patch, without this control.
#pragma once
#include <cstdlib>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace glm_mc2_diag {
inline unsigned char mode() {
    static const unsigned char *value = []() -> const unsigned char * {
        const char *path = std::getenv("GLM_MC2_CAPABILITY_MODE_FILE");
        if (!path) return nullptr;
        int fd = ::open(path, O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
        TORCH_CHECK(fd >= 0, "Cannot open MC2 diagnostic mode file");
        struct stat st;
        bool valid = ::fstat(fd, &st) == 0 && S_ISREG(st.st_mode) && st.st_size == 1;
        if (!valid) ::close(fd);
        TORCH_CHECK(valid, "Invalid MC2 diagnostic mode file");
        void *mapping = ::mmap(nullptr, 1, PROT_READ, MAP_SHARED, fd, 0);
        ::close(fd);
        TORCH_CHECK(mapping != MAP_FAILED, "Cannot map MC2 diagnostic mode file");
        return static_cast<const unsigned char *>(mapping);
    }();
    unsigned char current = value ? __atomic_load_n(value, __ATOMIC_ACQUIRE) : 0;
    TORCH_CHECK(current <= 1, "MC2 mode must be stock0 or cached1");
    return current;
}
inline bool dispatch_available() {
    static const bool available = check_aclnn_kernel_available("aclnnMoeDistributeDispatchV4");
    return mode() == 1 ? available : check_aclnn_kernel_available("aclnnMoeDistributeDispatchV4");
}
inline bool combine_available() {
    static const bool available = check_aclnn_kernel_available("aclnnMoeDistributeCombineV4");
    return mode() == 1 ? available : check_aclnn_kernel_available("aclnnMoeDistributeCombineV4");
}
}  // namespace glm_mc2_diag
