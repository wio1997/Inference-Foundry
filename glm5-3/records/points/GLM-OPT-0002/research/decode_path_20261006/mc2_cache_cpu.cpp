// Host capability predicate only; never creates a tensor or NPU context.
#include <atomic>
#include <chrono>
#include <cstdint>
#include <string>
#include <thread>
#include <vector>

using Lookup = void *(*)(const char *);
static Lookup actual_lookup = nullptr;

// Exact fixed-version check_aclnn_kernel_available predicate body.
static bool uncached(std::string name) {
    std::string workspace_name = name + "GetWorkspaceSize";
    if (actual_lookup(name.c_str()) == nullptr ||
        actual_lookup(workspace_name.c_str()) == nullptr) return false;
    return true;
}

template <int Op> static bool cached() {
    static const bool available = uncached(Op == 0 ?
        "aclnnMoeDistributeDispatchV4" : "aclnnMoeDistributeCombineV4");
    return available;
}

extern "C" int set_lookup(uintptr_t address) {
    if (!address || actual_lookup) return -1;
    actual_lookup = reinterpret_cast<Lookup>(address);
    return 0;
}

extern "C" uint64_t run_predicate(int op, int use_cache, int count,
                                   uint64_t *elapsed_ns) {
    if (!actual_lookup || op < 0 || op > 1 || count <= 0) return UINT64_MAX;
    const char *name = op == 0 ? "aclnnMoeDistributeDispatchV4" :
                                "aclnnMoeDistributeCombineV4";
    uint64_t success = 0;
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < count; ++i) {
        bool value = use_cache ? (op == 0 ? cached<0>() : cached<1>()) :
                                uncached(name);
        asm volatile("" : "+r"(value) : : "memory");
        success += value;
    }
    *elapsed_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - start).count();
    return success;
}

// Concurrent first-call, true and both short-circuit false outcomes.
template <int Case> struct Mock {
    static std::atomic<int> calls;
    static void *lookup(const char *name) {
        ++calls;
        bool workspace = std::string(name).find("GetWorkspaceSize") !=
                         std::string::npos;
        if (Case == 1 || (Case == 2 && workspace)) return nullptr;
        return reinterpret_cast<void *>(uintptr_t(1));
    }
    static bool check() {
        static const bool answer = []() {
            std::string name = "mock", workspace_name = name + "GetWorkspaceSize";
            if (lookup(name.c_str()) == nullptr ||
                lookup(workspace_name.c_str()) == nullptr) return false;
            return true;
        }();
        return answer;
    }
};
template <int Case> std::atomic<int> Mock<Case>::calls{0};

extern "C" int correctness() {
    std::atomic<int> failures{0};
    std::vector<std::thread> threads;
    for (int t = 0; t < 8; ++t) threads.emplace_back([&]() {
        for (int i = 0; i < 1000; ++i)
            if (!Mock<0>::check() || Mock<1>::check() || Mock<2>::check())
                ++failures;
    });
    for (auto &thread : threads) thread.join();
    return failures == 0 && Mock<0>::calls == 2 && Mock<1>::calls == 1 &&
           Mock<2>::calls == 2 ? 0 : 1;
}
