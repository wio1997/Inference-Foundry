#pragma once

#include <memory>
#include <torch_npu/csrc/core/npu/NPUEvent.h>

#include "tensor.hpp"

namespace deep_ep {

struct EventHandle {
    std::shared_ptr<c10_npu::NPUEvent> event;

    EventHandle(): EventHandle(c10_npu::getCurrentNPUStream()) {}

    explicit EventHandle(const c10_npu::NPUStream& stream):
        event(std::make_shared<c10_npu::NPUEvent>()) {
        event->record(stream);
    }

    EventHandle(const EventHandle& other) = default;

    void current_stream_wait() const {
        event->block(c10_npu::getCurrentNPUStream());
    }
};

}  // namespace deep_ep
