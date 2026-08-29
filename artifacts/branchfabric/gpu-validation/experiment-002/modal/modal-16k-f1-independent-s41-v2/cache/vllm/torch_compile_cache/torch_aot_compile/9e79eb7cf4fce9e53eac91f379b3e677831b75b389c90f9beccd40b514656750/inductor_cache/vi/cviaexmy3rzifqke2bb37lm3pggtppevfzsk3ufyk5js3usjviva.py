r"""
Compile-time auto-tuning block: 

import torch
from torch._dynamo.testing import rand_strided
from torch._dynamo.utils import preserve_rng_state
from torch._inductor.select_algorithm import AlgorithmSelectorCache
from torch._inductor.async_compile import AsyncCompile

async_compile = AsyncCompile()
generate_example_value = AlgorithmSelectorCache.generate_example_value
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
get_raw_stream = torch._C._cuda_getCurrentRawStream


# kernel path: /tmp/sloforge-branchfabric/modal-16k-f1-independent-s41-v2/cache/vllm/torch_compile_cache/torch_aot_compile/9e79eb7cf4fce9e53eac91f379b3e677831b75b389c90f9beccd40b514656750/inductor_cache/fo/cfosgltmk6phth6xocthp4logxvrld5daamhi2vqbqm37ux4vp2l.py
# Topologically Sorted Source Nodes: [long, embedding, rms_norm_default], Original ATen: [aten._to_copy, aten.embedding, vllm_ir.rms_norm]
# Source node to ATen node mapping:
#   embedding => embedding
#   long => convert_element_type
#   rms_norm_default => add_tensor, convert_element_type_default, convert_element_type_default_1, mean_dim, mul_tensor, mul_tensor_1, pow_tensor_scalar, rsqrt_default
# Graph fragment:
#   %arg0_1 : Tensor "i32[s72][1]cuda:0" = PlaceHolder[target=arg0_1]
#   %arg2_1 : Tensor "bf16[152064, 3584][3584, 1]cuda:0" = PlaceHolder[target=arg2_1]
#   %embedding : Tensor "bf16[s72, 3584][3584, 1]cuda:0" = PlaceHolder[target=embedding]
#   %buf1 : Tensor "f32[s72, 1][1, s72]cuda:0" = PlaceHolder[target=buf1]
#   %arg3_1 : Tensor "bf16[3584][1]cuda:0" = PlaceHolder[target=arg3_1]
#   %convert_element_type : Tensor "i64[s72][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg0_1, torch.int64), kwargs = {})
#   %embedding : Tensor "bf16[s72, 3584][3584, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.embedding.default](args = (%arg2_1, %convert_element_type), kwargs = {})
#   %convert_element_type_default : Tensor "f32[s72, 3584][3584, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%embedding, torch.float32), kwargs = {})
#   %pow_tensor_scalar : Tensor "f32[s72, 3584][3584, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_default, 2), kwargs = {})
#   %mean_dim : Tensor "f32[s72, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_tensor_scalar, [-1], True), kwargs = {})
#   %add_tensor : Tensor "f32[s72, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_dim, 1e-06), kwargs = {})
#   %rsqrt_default : Tensor "f32[s72, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_tensor,), kwargs = {})
#   %mul_tensor : Tensor "f32[s72, 3584][3584, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_default, %rsqrt_default), kwargs = {})
#   %convert_element_type_default_1 : Tensor "bf16[s72, 3584][3584, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_tensor, torch.bfloat16), kwargs = {})
#   %mul_tensor_1 : Tensor "bf16[s72, 3584][3584, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_default_1, %arg3_1), kwargs = {})
#   return %embedding,%buf1,%mul_tensor_1
triton_red_fused__to_copy_embedding_rms_norm_0 = async_compile.triton('triton_red_fused__to_copy_embedding_rms_norm_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 8192, 'r0_': 4096},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i32', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'out_ptr0': '*bf16', 'out_ptr2': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_embedding_rms_norm_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 3, 'num_store': 2, 'num_reduction': 1, 'backend_hash': 'D221555120843152094FD7E43096F924AB4102579828DDF622EEF3389D4AEF7F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 32768, 'r0_': 234888192}}
)
@triton.jit
def triton_red_fused__to_copy_embedding_rms_norm_0(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    r0_numel = 3584
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    _tmp11 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp1 = tmp0.to(tl.int64)
        tmp2 = tl.full([1, 1], 152064, tl.int32)
        tmp3 = tmp1 + tmp2
        tmp4 = tmp1 < 0
        tmp5 = tl.where(tmp4, tmp3, tmp1)
        tl.device_assert(((0 <= tmp5) & (tmp5 < 152064)) | ~(xmask), "index out of bounds: 0 <= tmp5 < 152064")
        tmp7 = tl.load(in_ptr1 + (r0_1 + 3584*tmp5), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp8 = tmp7.to(tl.float32)
        tmp9 = tmp8 * tmp8
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, R0_BLOCK])
        tmp12 = _tmp11 + tmp10
        _tmp11 = tl.where(r0_mask & xmask, tmp12, _tmp11)
        tl.store(out_ptr0 + (r0_1 + 3584*x0), tmp7, r0_mask & xmask)
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp13 = tl.load(out_ptr0 + (r0_1 + 3584*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp22 = tl.load(in_ptr2 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp14 = tmp13.to(tl.float32)
        tmp15 = tl.full([1, 1], 3584.0, tl.float32)
        tmp16 = (tmp11 / tmp15)
        tmp17 = tl.full([1, 1], 1e-06, tl.float32)
        tmp18 = tmp16 + tmp17
        tmp19 = libdevice.rsqrt(tmp18)
        tmp20 = tmp14 * tmp19
        tmp21 = tmp20.to(tl.float32)
        tmp23 = tmp21 * tmp22
        tl.store(out_ptr2 + (r0_1 + 3584*x0), tmp23, r0_mask & xmask)
''', device_str='cuda')


# kernel path: /tmp/sloforge-branchfabric/modal-16k-f1-independent-s41-v2/cache/vllm/torch_compile_cache/torch_aot_compile/9e79eb7cf4fce9e53eac91f379b3e677831b75b389c90f9beccd40b514656750/inductor_cache/7v/c7vflzhcnzyjmsrflh7yca5r4lgl74b6lu4ckhzxc5ytzslse4b7.py
# Unsorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
triton_poi_fused_1 = async_compile.triton('triton_poi_fused_1', '''
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
    inductor_meta={'grid_type': 'SequentialComboKernelGrid', 'combo_grid_meta': {'num_kernels': 2, 'min_blocks': None, 'default_config': None, 'no_x_dim_0': False, 'xnumel_0': None, 'no_x_dim_1': False, 'xnumel_1': None}, 'kernel_name': 'triton_poi_fused_1', 'mutated_arg_names': [], 'backend_hash': 'D221555120843152094FD7E43096F924AB4102579828DDF622EEF3389D4AEF7F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def triton_poi_fused_1(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel_0, xnumel_1, XBLOCK : tl.constexpr):
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
        triton_poi_fused_1.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused_1.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(call, fn_args=(args,), device=cuda,rep=40)
    num_gb = 0
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')

async_compile.wait(globals())
del async_compile

import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
with torch.cuda._DeviceGuard(0):
    stream0 = get_raw_stream(0)
stream0 = get_raw_stream(0)
arg0_1 = generate_example_value((8192,), (1,), 'cuda:0', torch.int32, 0, (8192,))
arg2_1 = generate_example_value((152064, 3584), (3584, 1), 'cuda:0', torch.bfloat16, 0, (152064, 3584))
arg3_1 = generate_example_value((3584,), (1,), 'cuda:0', torch.bfloat16, 0, (3584,))
buf0 = generate_example_value((8192, 3584), (3584, 1), 'cuda:0', torch.bfloat16, 0, (8192, 3584))
buf2 = generate_example_value((8192, 3584), (3584, 1), 'cuda:0', torch.bfloat16, 0, (8192, 3584))
with torch.cuda._DeviceGuard(0):
    triton_red_fused__to_copy_embedding_rms_norm_0.run(arg0_1, arg2_1, arg3_1, buf0, buf2, 8192, 3584, stream=stream0)
del arg0_1, arg2_1, arg3_1, buf0, buf2

stream0 = get_raw_stream(0)
buf3 = generate_example_value((8192, 4608), (4608, 1), 'cuda:0', torch.bfloat16, 0, (8192, 4608))
arg6_1 = generate_example_value((8192,), (1,), 'cuda:0', torch.int64, 0, (8192,))
arg7_1 = generate_example_value((32768, 128), (128, 1), 'cuda:0', torch.bfloat16, 0, (32768, 128))
buf4 = generate_example_value((8192, 4, 128), (512, 128, 1), 'cuda:0', torch.bfloat16, 0, (8192, 4, 128))
buf5 = generate_example_value((8192, 28, 128), (3584, 128, 1), 'cuda:0', torch.bfloat16, 0, (8192, 28, 128))
with torch.cuda._DeviceGuard(0):
    triton_poi_fused_1.run(buf3, arg6_1, arg7_1, buf4, buf5, 4194304, 29360128, stream=stream0)
del buf3, arg6_1, arg7_1, buf4, buf5

"""
# AOT ID: ['0_inference']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from cmath import nanj
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_raw_stream

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
assert_alignment = torch._C._dynamo.guards.assert_alignment
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cpu_pinned = torch._C._dynamo.guards._empty_strided_cpu_pinned
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
empty_strided_mtia = torch._C._dynamo.guards._empty_strided_mtia
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()
empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p


# kernel path: /tmp/sloforge-branchfabric/modal-16k-f1-independent-s41-v2/cache/vllm/torch_compile_cache/torch_aot_compile/9e79eb7cf4fce9e53eac91f379b3e677831b75b389c90f9beccd40b514656750/inductor_cache/fo/cfosgltmk6phth6xocthp4logxvrld5daamhi2vqbqm37ux4vp2l.py
# Topologically Sorted Source Nodes: [long, embedding, rms_norm_default], Original ATen: [aten._to_copy, aten.embedding, vllm_ir.rms_norm]
# Source node to ATen node mapping:
#   embedding => embedding
#   long => convert_element_type
#   rms_norm_default => add_tensor, convert_element_type_default, convert_element_type_default_1, mean_dim, mul_tensor, mul_tensor_1, pow_tensor_scalar, rsqrt_default
# Graph fragment:
#   %arg0_1 : Tensor "i32[s72][1]cuda:0" = PlaceHolder[target=arg0_1]
#   %arg2_1 : Tensor "bf16[152064, 3584][3584, 1]cuda:0" = PlaceHolder[target=arg2_1]
#   %embedding : Tensor "bf16[s72, 3584][3584, 1]cuda:0" = PlaceHolder[target=embedding]
#   %buf1 : Tensor "f32[s72, 1][1, s72]cuda:0" = PlaceHolder[target=buf1]
#   %arg3_1 : Tensor "bf16[3584][1]cuda:0" = PlaceHolder[target=arg3_1]
#   %convert_element_type : Tensor "i64[s72][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg0_1, torch.int64), kwargs = {})
#   %embedding : Tensor "bf16[s72, 3584][3584, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.embedding.default](args = (%arg2_1, %convert_element_type), kwargs = {})
#   %convert_element_type_default : Tensor "f32[s72, 3584][3584, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%embedding, torch.float32), kwargs = {})
#   %pow_tensor_scalar : Tensor "f32[s72, 3584][3584, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_default, 2), kwargs = {})
#   %mean_dim : Tensor "f32[s72, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_tensor_scalar, [-1], True), kwargs = {})
#   %add_tensor : Tensor "f32[s72, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_dim, 1e-06), kwargs = {})
#   %rsqrt_default : Tensor "f32[s72, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_tensor,), kwargs = {})
#   %mul_tensor : Tensor "f32[s72, 3584][3584, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_default, %rsqrt_default), kwargs = {})
#   %convert_element_type_default_1 : Tensor "bf16[s72, 3584][3584, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_tensor, torch.bfloat16), kwargs = {})
#   %mul_tensor_1 : Tensor "bf16[s72, 3584][3584, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_default_1, %arg3_1), kwargs = {})
#   return %embedding,%buf1,%mul_tensor_1
triton_red_fused__to_copy_embedding_rms_norm_0 = async_compile.triton('triton_red_fused__to_copy_embedding_rms_norm_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 8192, 'r0_': 4096},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i32', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'out_ptr0': '*bf16', 'out_ptr2': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_embedding_rms_norm_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 3, 'num_store': 2, 'num_reduction': 1, 'backend_hash': 'D221555120843152094FD7E43096F924AB4102579828DDF622EEF3389D4AEF7F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 32768, 'r0_': 234888192}}
)
@triton.jit
def triton_red_fused__to_copy_embedding_rms_norm_0(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    r0_numel = 3584
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    _tmp11 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp1 = tmp0.to(tl.int64)
        tmp2 = tl.full([1, 1], 152064, tl.int32)
        tmp3 = tmp1 + tmp2
        tmp4 = tmp1 < 0
        tmp5 = tl.where(tmp4, tmp3, tmp1)
        tl.device_assert(((0 <= tmp5) & (tmp5 < 152064)) | ~(xmask), "index out of bounds: 0 <= tmp5 < 152064")
        tmp7 = tl.load(in_ptr1 + (r0_1 + 3584*tmp5), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp8 = tmp7.to(tl.float32)
        tmp9 = tmp8 * tmp8
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, R0_BLOCK])
        tmp12 = _tmp11 + tmp10
        _tmp11 = tl.where(r0_mask & xmask, tmp12, _tmp11)
        tl.store(out_ptr0 + (r0_1 + 3584*x0), tmp7, r0_mask & xmask)
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp13 = tl.load(out_ptr0 + (r0_1 + 3584*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp22 = tl.load(in_ptr2 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp14 = tmp13.to(tl.float32)
        tmp15 = tl.full([1, 1], 3584.0, tl.float32)
        tmp16 = (tmp11 / tmp15)
        tmp17 = tl.full([1, 1], 1e-06, tl.float32)
        tmp18 = tmp16 + tmp17
        tmp19 = libdevice.rsqrt(tmp18)
        tmp20 = tmp14 * tmp19
        tmp21 = tmp20.to(tl.float32)
        tmp23 = tmp21 * tmp22
        tl.store(out_ptr2 + (r0_1 + 3584*x0), tmp23, r0_mask & xmask)
''', device_str='cuda')


# kernel path: /tmp/sloforge-branchfabric/modal-16k-f1-independent-s41-v2/cache/vllm/torch_compile_cache/torch_aot_compile/9e79eb7cf4fce9e53eac91f379b3e677831b75b389c90f9beccd40b514656750/inductor_cache/7v/c7vflzhcnzyjmsrflh7yca5r4lgl74b6lu4ckhzxc5ytzslse4b7.py
# Unsorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
triton_poi_fused_1 = async_compile.triton('triton_poi_fused_1', '''
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
    inductor_meta={'grid_type': 'SequentialComboKernelGrid', 'combo_grid_meta': {'num_kernels': 2, 'min_blocks': None, 'default_config': None, 'no_x_dim_0': False, 'xnumel_0': None, 'no_x_dim_1': False, 'xnumel_1': None}, 'kernel_name': 'triton_poi_fused_1', 'mutated_arg_names': [], 'backend_hash': 'D221555120843152094FD7E43096F924AB4102579828DDF622EEF3389D4AEF7F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def triton_poi_fused_1(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel_0, xnumel_1, XBLOCK : tl.constexpr):
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
        triton_poi_fused_1.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused_1.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(call, fn_args=(args,), device=cuda,rep=40)
    num_gb = 0
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

class Runner:
    def __init__(self, partitions):
        self.partitions = partitions

    def recursively_apply_fns(self, fns):
        new_callables = []
        for fn, c in zip(fns, self.partitions):
            new_callables.append(fn(c))
        self.partitions = new_callables

    def call(self, args):
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1 = args
        args.clear()
        s72 = arg1_1
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            buf0 = empty_strided_cuda((s72, 3584), (3584, 1), torch.bfloat16)
            buf2 = empty_strided_cuda((s72, 3584), (3584, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [long, embedding, rms_norm_default], Original ATen: [aten._to_copy, aten.embedding, vllm_ir.rms_norm]
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_embedding_rms_norm_0.run(arg0_1, arg2_1, arg3_1, buf0, buf2, s72, 3584, stream=stream0)
            del arg0_1
            del arg2_1
            del arg3_1
            buf3 = empty_strided_cuda((s72, 4608), (4608, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [rms_norm_default, linear], Original ATen: [vllm_ir.rms_norm, aten.t, aten.addmm]
            extern_kernels.addmm(arg5_1, buf2, reinterpret_tensor(arg4_1, (3584, 4608), (1, 3584), 0), alpha=1, beta=1, out=buf3)
            del arg4_1
            del arg5_1
            buf4 = empty_strided_cuda((s72, 4, 128), (512, 128, 1), torch.bfloat16)
            buf5 = reinterpret_tensor(buf2, (s72, 28, 128), (3584, 128, 1), 0); del buf2  # reuse
            # Topologically Sorted Source Nodes: [split, index_select, chunk, view, chunk_1, view_1, chunk_2, unsqueeze_2, mul_4, unsqueeze_3, mul_5, sub_1, mul_6, mul_7, add_1, cat_2, unsqueeze, mul, unsqueeze_1, mul_1, sub, mul_2, mul_3, add, cat], Original ATen: [aten.split_with_sizes, aten.index_select, aten.split, aten.view, aten.unsqueeze, aten.mul, aten.sub, aten.add, aten.cat]
            triton_poi_fused_1_xnumel_0 = 512*s72
            triton_poi_fused_1_xnumel_1 = 3584*s72
            stream0 = get_raw_stream(0)
            triton_poi_fused_1.run(buf3, arg6_1, arg7_1, buf4, buf5, triton_poi_fused_1_xnumel_0, triton_poi_fused_1_xnumel_1, stream=stream0)
            del arg6_1
            del arg7_1
            buf6 = empty_strided_cuda((s72, 3584), (3584, 1), torch.bfloat16)
        return (buf4, reinterpret_tensor(buf3, (s72, 4, 128), (4608, 128, 1), 4096), buf5, reinterpret_tensor(buf6, (s72, 28, 128), (3584, 128, 1), 0), buf0, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    arg0_1 = rand_strided((8192, ), (1, ), device='cuda:0', dtype=torch.int32)
    arg1_1 = 8192
    arg2_1 = rand_strided((152064, 3584), (3584, 1), device='cuda:0', dtype=torch.bfloat16)
    arg3_1 = rand_strided((3584, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg4_1 = rand_strided((4608, 3584), (3584, 1), device='cuda:0', dtype=torch.bfloat16)
    arg5_1 = rand_strided((4608, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg6_1 = rand_strided((8192, ), (1, ), device='cuda:0', dtype=torch.int64)
    arg7_1 = rand_strided((32768, 128), (128, 1), device='cuda:0', dtype=torch.bfloat16)
    return [arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('None', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
