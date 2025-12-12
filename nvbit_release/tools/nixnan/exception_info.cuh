#ifndef EXCEPTION_INFO_HH
#define EXCEPTION_INFO_HH
#include <vector_types.h>
#include <stdint.h>
#define WARPSIZE 32
namespace nixnan {
    struct exception_info {
        // Constructor for regular exceptions (NaN/Inf/etc)
        __host__ __device__
        exception_info(int4 cta, uint32_t warp_id, uint32_t inst_id, uint32_t exce, uint32_t operand,
                       uint32_t tp_override = UNKNOWN, bool skip = false) : skip(skip),
                       inst_id(inst_id), exce(exce), oper(operand), tp_override(tp_override),
                       is_barrier(false), barrier_count(0) {
            this->cta = cta;
            cta.w = warp_id;
        }

        // Constructor for barrier events
        __host__ __device__
        exception_info(int4 cta, uint32_t warp_id, uint32_t inst_id, uint32_t barrier_cnt)
                       : skip(false), inst_id(inst_id), exce(0), oper(0), tp_override(UNKNOWN),
                       is_barrier(true), barrier_count(barrier_cnt) {
            this->cta = cta;
            cta.w = warp_id;
        }
        __host__ __device__ inline int x() const {
            return cta.x;
        }
        __host__ __device__ inline int y() const {
            return cta.y;
        }
        __host__ __device__ inline int z() const {
            return cta.z;
        }
        __host__ __device__ inline int warp() const {
            return cta.w;
        }
        __host__ __device__ inline uint32_t inst() const {
            return inst_id;
        }
        __host__ __device__ inline uint32_t exception() const {
            return exce;
        }

        __host__ __device__ inline uint32_t operand() const {
            return oper;
        }
        
        __host__ __device__ inline uint32_t type() const {
            return tp_override;
        }
        __host__ __device__ inline bool to_skip() const {
            return skip;
        }

        __host__ __device__ inline bool is_barrier_event() const {
            return is_barrier;
        }

        __host__ __device__ inline uint32_t barrier_cnt() const {
            return barrier_count;
        }

    private:
        exception_info() {};
        int4 cta;
        uint32_t inst_id;
        uint32_t exce;
        uint32_t oper;
        uint32_t tp_override;
        bool skip = false;
        bool is_barrier = false;
        uint32_t barrier_count = 0;
    };
}
#undef WARPSIZE
#endif // EXCEPTION_INFO_HH