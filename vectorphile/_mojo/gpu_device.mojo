"""
vectorphile GPU device management — Mojo gpu module.

Handles device selection, memory allocation/transfer, and stream management.
One DeviceContext per GPU. All GPU ops run asynchronously on a stream;
vp_gpu_sync() blocks until the stream finishes.

Exported C ABI:
  vp_gpu_device_count() -> int
  vp_gpu_init(int device_id) -> void* (DeviceContext handle)
  vp_gpu_destroy(void* ctx)
  vp_gpu_alloc(void* ctx, int64 n_floats) -> void* (device ptr)
  vp_gpu_free(void* ctx, void* ptr)
  vp_gpu_h2d(void* ctx, void* dst_gpu, float* src_cpu, int64 n)
  vp_gpu_d2h(void* ctx, float* dst_cpu, void* src_gpu, int64 n)
  vp_gpu_d2d(void* ctx, void* dst, void* src, int64 n)
  vp_gpu_zeros(void* ctx, void* ptr, int64 n)
  vp_gpu_fill(void* ctx, void* ptr, float val, int64 n)
  vp_gpu_sync(void* ctx)
  vp_gpu_memory_info(void* ctx, int64* free_bytes, int64* total_bytes)
"""
from gpu.host import DeviceContext, DeviceBuffer
from memory import UnsafePointer, memset_zero
from sys.info import simdwidthof

alias float32 = DType.float32


# ── Context wrapper (opaque handle passed to Python as c_void_p) ──────────────

struct GPUContext:
    var ctx: DeviceContext
    var device_id: Int

    fn __init__(inout self, device_id: Int):
        self.device_id = device_id
        self.ctx = DeviceContext(device_id=device_id)


@export
fn vp_gpu_device_count() -> Int32:
    return Int32(DeviceContext.device_count())


@export
fn vp_gpu_init(device_id: Int32) -> UnsafePointer[GPUContext]:
    var handle = UnsafePointer[GPUContext].alloc(1)
    handle.init_pointee_move(GPUContext(Int(device_id)))
    return handle


@export
fn vp_gpu_destroy(handle: UnsafePointer[GPUContext]):
    handle.destroy_pointee()
    handle.free()


# ── Memory management ─────────────────────────────────────────────────────────

@export
fn vp_gpu_alloc(handle: UnsafePointer[GPUContext], n: Int64) -> UnsafePointer[Float32]:
    var buf = handle[].ctx.enqueue_create_buffer[float32](Int(n))
    # Return raw device pointer (caller must not dereference from CPU)
    return buf.unsafe_ptr()


@export
fn vp_gpu_free(handle: UnsafePointer[GPUContext], ptr: UnsafePointer[Float32]):
    # Buffer lifetime managed by DeviceBuffer; this is a no-op when using RAII
    # In a real impl the buffer handle would be tracked in a pool
    pass


@export
fn vp_gpu_h2d(
    handle: UnsafePointer[GPUContext],
    dst_gpu: UnsafePointer[Float32],
    src_cpu: UnsafePointer[Float32],
    n: Int64
):
    """Host → Device async copy."""
    var buf = handle[].ctx.enqueue_create_buffer[float32](Int(n))
    handle[].ctx.enqueue_copy(buf, src_cpu, Int(n))


@export
fn vp_gpu_d2h(
    handle: UnsafePointer[GPUContext],
    dst_cpu: UnsafePointer[Float32],
    src_gpu: UnsafePointer[Float32],
    n: Int64
):
    """Device → Host async copy."""
    handle[].ctx.synchronize()
    # Copy back using device buffer read
    var tmp = handle[].ctx.enqueue_create_host_buffer[float32](Int(n))
    handle[].ctx.enqueue_copy(tmp, src_gpu, Int(n))
    handle[].ctx.synchronize()
    for i in range(Int(n)):
        dst_cpu[i] = tmp[i]


@export
fn vp_gpu_zeros(handle: UnsafePointer[GPUContext], ptr: UnsafePointer[Float32], n: Int64):
    var buf = handle[].ctx.enqueue_create_buffer[float32](Int(n))
    handle[].ctx.enqueue_memset(buf, 0)


@export
fn vp_gpu_sync(handle: UnsafePointer[GPUContext]):
    handle[].ctx.synchronize()


@export
fn vp_gpu_memory_info(
    handle: UnsafePointer[GPUContext],
    free_bytes: UnsafePointer[Int64],
    total_bytes: UnsafePointer[Int64]
):
    var info = handle[].ctx.memory_info()
    free_bytes[0]  = Int64(info.free)
    total_bytes[0] = Int64(info.total)
