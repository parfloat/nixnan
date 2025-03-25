#ifndef EXCEPTION_INFO_HH
#define EXCEPTION_INFO_HH
#include <vector_types.h>
#include <stdint.h>
#define WARPSIZE 32
namespace nixnan {
    struct exception_info {
        uint32_t exceptions[WARPSIZE];
        __host__ __device__
        exception_info(int4 cta, uint32_t warp_id, uint32_t inst_id, uint32_t exce) :
                       inst_id(inst_id), exce(exce) {
            this->cta = cta;
            cta.w = warp_id;
        }
        __host__ __device__ int x() const {
            return cta.x;
        }
        __host__ __device__ int y() const {
            return cta.y;
        }
        __host__ __device__ int z() const {
            return cta.z;
        }
        __host__ __device__ int warp() const {
            return cta.w;
        }
        __host__ __device__ uint32_t inst() const {
            return inst_id;
        }
        __host__ __device__ uint32_t exception() const {
            return exce;
        }
    private:
        exception_info() {};
        int4 cta;
        uint32_t inst_id;
        uint32_t exce;
    };
}
#undef WARPSIZE
#endif // EXCEPTION_INFO_HH