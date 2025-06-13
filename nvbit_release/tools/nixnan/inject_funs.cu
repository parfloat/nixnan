#include "fp_utils.cuh"
#include <cstdint>
#include "utils/utils.h"
#include <cstdarg>
#include "exception_info.cuh"
#include "recording.h"
#include "utils/channel.hpp"
#include "reginfo.cuh"

using namespace nixnan;

extern "C" __device__ __noinline__ void
nixnan_check_regs(int pred, device_recorder recorder, uint32_t inst_id,
                  ChannelDev* pchannel_dev, uint32_t n_args...) {

  if (!pred) {
    return;
  }

  int active_mask = __ballot_sync(__activemask(), 1);
  const int laneid = get_laneid();
  const int first_laneid = __ffs(active_mask) - 1;
  va_list ap;
  va_start(ap, n_args);
  uint32_t exce = 0;
  for (int i = 0; i < n_args; i++) {
    reginfo reg_info = va_arg(ap, reginfo);
    for (int j = 0; j < reg_info.count; j++) {
      switch (reg_info.type) {
        case FP16: {
          uint32_t val = va_arg(ap, uint32_t);
          exce |= half2_classify(val, reg_info.half_h0, reg_info.half_h1);
          break;
        }
        case BF16: {
          uint32_t val = va_arg(ap, uint32_t);
          exce |= half2_classify(val, reg_info.half_h0, reg_info.half_h1);
          break;
        }
        case FP32: {
          uint32_t val = va_arg(ap, uint32_t);
          exce |= float_classify(val);
          break;
        }
        case FP64: {
          uint32_t low = va_arg(ap, uint32_t);
          uint32_t high = va_arg(ap, uint32_t);
          j++;
          exce |= double_classify(low, high);
          break;
        }
        default:
          // assert(0 && "Unknown type");
          break;
      }
    }
  }
  va_end(ap);

  exception_info ei{get_ctaid(), get_warpid(), inst_id, exce};
  for (int tid = 0; tid < 32; tid++) {
    exce |= __shfl_sync(active_mask, exce, tid);
  }

  /* first active lane pushes information on the channel */
  if (first_laneid == laneid) {
    if (exce != 0) {
      uint32_t num_exceptions = recorder.record(inst_id, exce);
      if (num_exceptions == 0) {
        ChannelDev *channel_dev = (ChannelDev *)pchannel_dev;
        channel_dev->push(&ei, sizeof(exception_info));
      }
    }
  }
}