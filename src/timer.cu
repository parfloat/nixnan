#include "timer.h"
#include <csignal>
#include <thread>

namespace nixnan{
namespace timer {
    size_t timeout = 0;
    std::thread timeout_thread;
    void init() {
        GET_VAR_INT(timeout, "NIXNAN_TIMEOUT", 0,
        "Timeout for the nixnan instrumented process in seconds.")
    }

    void tool_init(CUcontext ctx) {
        if (timeout > 0) {
            timeout_thread = std::thread([=]() {
                std::this_thread::sleep_for(std::chrono::seconds(timeout));
                std::raise(SIGINT);
            });
        }
    }

    void instrument(CUcontext ctx, Instr* instr, const std::string& kname) {
        return;
    }

    void term() {
        if (timeout_thread.joinable()) {
            timeout_thread.join();
        }
        return;
    }
}
}