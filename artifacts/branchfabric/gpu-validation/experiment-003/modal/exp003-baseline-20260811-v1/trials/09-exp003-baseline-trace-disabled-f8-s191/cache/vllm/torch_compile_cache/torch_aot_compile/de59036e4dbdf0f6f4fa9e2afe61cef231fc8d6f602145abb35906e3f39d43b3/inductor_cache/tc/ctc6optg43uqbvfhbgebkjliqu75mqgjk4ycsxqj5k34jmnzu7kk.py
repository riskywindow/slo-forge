
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

from torch._dynamo.testing import rand_strided
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch

@triton_heuristics.pointwise(
    size_hints={'x': 33554432}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*i64', 'in_ptr2': '*bf16', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'xnumel_0': 'i32', 'xnumel_1': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'enable_fp_fusion': True, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'SequentialComboKernelGrid', 'combo_grid_meta': {'num_kernels': 2, 'min_blocks': None, 'default_config': None, 'no_x_dim_0': False, 'xnumel_0': None, 'no_x_dim_1': False, 'xnumel_1': None}, 'kernel_name': 'triton_poi_fused_3', 'mutated_arg_names': [], 'backend_hash': 'D221555120843152094FD7E43096F924AB4102579828DDF622EEF3389D4AEF7F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def triton_poi_fused_3(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel_0, xnumel_1, XBLOCK : tl.constexpr):
    pid = tl.program_id(0)
    num_xblocks_0 = tl.cdiv(xnumel_0, XBLOCK)
    num_xblocks_1 = num_xblocks_0 + tl.cdiv(xnumel_1, XBLOCK)
    if pid < num_xblocks_0:
        pid_offset = pid
        r0_numel = 1
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel_0
        x0 = (xindex % 128)
        x1 = ((xindex // 128) % 4)
        x2 = xindex // 512
        x4 = xindex
        tmp0 = x0
        tmp1 = tl.full([1], 0, tl.int64)
        tmp2 = tmp0 >= tmp1
        tmp3 = tl.full([1], 64, tl.int64)
        tmp4 = tmp0 < tmp3
        tmp5 = tl.load(in_ptr0 + (3584 + 128*x1 + 4608*x2 + (x0)), tmp4 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp6 = tl.load(in_ptr1 + (x2), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
        tmp7 = tl.full([XBLOCK], 32768, tl.int32)
        tmp8 = tmp6 + tmp7
        tmp9 = tmp6 < 0
        tmp10 = tl.where(tmp9, tmp8, tmp6)
        tl.device_assert(((0 <= tl.broadcast_to(tmp10, [XBLOCK])) & (tl.broadcast_to(tmp10, [XBLOCK]) < 32768)) | ~(tmp4 & xmask), "index out of bounds: 0 <= tl.broadcast_to(tmp10, [XBLOCK]) < 32768")
        tmp12 = tl.load(in_ptr2 + (128*tmp10 + (x0)), tmp4 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp13 = tmp5 * tmp12
        tmp14 = tl.load(in_ptr0 + (3648 + 128*x1 + 4608*x2 + (x0)), tmp4 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp15 = tl.load(in_ptr2 + (64 + 128*tmp10 + (x0)), tmp4 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp16 = tmp14 * tmp15
        tmp17 = tmp13 - tmp16
        tmp18 = tl.full(tmp17.shape, 0.0, tmp17.dtype)
        tmp19 = tl.where(tmp4, tmp17, tmp18)
        tmp20 = tmp0 >= tmp3
        tmp21 = tl.full([1], 128, tl.int64)
        tmp22 = tmp0 < tmp21
        tmp23 = tl.load(in_ptr0 + (3648 + 128*x1 + 4608*x2 + ((-64) + x0)), tmp20 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp24 = tl.load(in_ptr1 + (x2), tmp20 & xmask, eviction_policy='evict_last', other=0.0)
        tmp25 = tl.full([XBLOCK], 32768, tl.int32)
        tmp26 = tmp24 + tmp25
        tmp27 = tmp24 < 0
        tmp28 = tl.where(tmp27, tmp26, tmp24)
        tl.device_assert(((0 <= tl.broadcast_to(tmp28, [XBLOCK])) & (tl.broadcast_to(tmp28, [XBLOCK]) < 32768)) | ~(tmp20 & xmask), "index out of bounds: 0 <= tl.broadcast_to(tmp28, [XBLOCK]) < 32768")
        tmp30 = tl.load(in_ptr2 + (128*tmp28 + ((-64) + x0)), tmp20 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp31 = tmp23 * tmp30
        tmp32 = tl.load(in_ptr0 + (3584 + 128*x1 + 4608*x2 + ((-64) + x0)), tmp20 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp33 = tl.load(in_ptr2 + (64 + 128*tmp28 + ((-64) + x0)), tmp20 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp34 = tmp32 * tmp33
        tmp35 = tmp31 + tmp34
        tmp36 = tl.full(tmp35.shape, 0.0, tmp35.dtype)
        tmp37 = tl.where(tmp20, tmp35, tmp36)
        tmp38 = tl.where(tmp4, tmp19, tmp37)
        tl.store(out_ptr0 + (x4), tmp38, xmask)
    elif pid < num_xblocks_1:
        pid_offset = pid - num_xblocks_0
        r0_numel = 1
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel_1
        x5 = (xindex % 128)
        x6 = ((xindex // 128) % 28)
        x7 = xindex // 3584
        x9 = xindex
        tmp39 = x5
        tmp40 = tl.full([1], 0, tl.int64)
        tmp41 = tmp39 >= tmp40
        tmp42 = tl.full([1], 64, tl.int64)
        tmp43 = tmp39 < tmp42
        tmp44 = tl.load(in_ptr0 + (128*x6 + 4608*x7 + (x5)), tmp43 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp45 = tl.load(in_ptr1 + (x7), tmp43 & xmask, eviction_policy='evict_last', other=0.0)
        tmp46 = tl.full([XBLOCK], 32768, tl.int32)
        tmp47 = tmp45 + tmp46
        tmp48 = tmp45 < 0
        tmp49 = tl.where(tmp48, tmp47, tmp45)
        tl.device_assert(((0 <= tl.broadcast_to(tmp49, [XBLOCK])) & (tl.broadcast_to(tmp49, [XBLOCK]) < 32768)) | ~(tmp43 & xmask), "index out of bounds: 0 <= tl.broadcast_to(tmp49, [XBLOCK]) < 32768")
        tmp51 = tl.load(in_ptr2 + (128*tmp49 + (x5)), tmp43 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp52 = tmp44 * tmp51
        tmp53 = tl.load(in_ptr0 + (64 + 128*x6 + 4608*x7 + (x5)), tmp43 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp54 = tl.load(in_ptr2 + (64 + 128*tmp49 + (x5)), tmp43 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp55 = tmp53 * tmp54
        tmp56 = tmp52 - tmp55
        tmp57 = tl.full(tmp56.shape, 0.0, tmp56.dtype)
        tmp58 = tl.where(tmp43, tmp56, tmp57)
        tmp59 = tmp39 >= tmp42
        tmp60 = tl.full([1], 128, tl.int64)
        tmp61 = tmp39 < tmp60
        tmp62 = tl.load(in_ptr0 + (64 + 128*x6 + 4608*x7 + ((-64) + x5)), tmp59 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp63 = tl.load(in_ptr1 + (x7), tmp59 & xmask, eviction_policy='evict_last', other=0.0)
        tmp64 = tl.full([XBLOCK], 32768, tl.int32)
        tmp65 = tmp63 + tmp64
        tmp66 = tmp63 < 0
        tmp67 = tl.where(tmp66, tmp65, tmp63)
        tl.device_assert(((0 <= tl.broadcast_to(tmp67, [XBLOCK])) & (tl.broadcast_to(tmp67, [XBLOCK]) < 32768)) | ~(tmp59 & xmask), "index out of bounds: 0 <= tl.broadcast_to(tmp67, [XBLOCK]) < 32768")
        tmp69 = tl.load(in_ptr2 + (128*tmp67 + ((-64) + x5)), tmp59 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp70 = tmp62 * tmp69
        tmp71 = tl.load(in_ptr0 + (128*x6 + 4608*x7 + ((-64) + x5)), tmp59 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp72 = tl.load(in_ptr2 + (64 + 128*tmp67 + ((-64) + x5)), tmp59 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp73 = tmp71 * tmp72
        tmp74 = tmp70 + tmp73
        tmp75 = tl.full(tmp74.shape, 0.0, tmp74.dtype)
        tmp76 = tl.where(tmp59, tmp74, tmp75)
        tmp77 = tl.where(tmp43, tmp58, tmp76)
        tl.store(out_ptr1 + (x9), tmp77, xmask)
    else:
        pass


def get_args():
    arg_0 = rand_strided((8192, 4608), (4608, 1), device='cuda:0', dtype=torch.bfloat16)
    arg_1 = rand_strided((8192,), (1,), device='cuda:0', dtype=torch.int64)
    arg_2 = rand_strided((32768, 128), (128, 1), device='cuda:0', dtype=torch.bfloat16)
    arg_3 = rand_strided((8192, 4, 128), (512, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    arg_4 = rand_strided((8192, 28, 128), (3584, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    return arg_0, arg_1, arg_2, arg_3, arg_4, 4194304, 29360128,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused_3.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused_3.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(call, fn_args=(args,), device=cuda,rep=40)
    num_gb = 0
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
