/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include "rasterizer_impl.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

#include "auxiliary.h"
#include "forward.h"
#include "backward.h"

// Helper function to find the next-highest bit of the MSB
// on the CPU.
// getHigherMsb 函数的作用是找到大于或等于给定数字的最高有效位 (Most Significant Bit, MSB) 的下一个较高位。
// 这在确定排序位宽时非常有用，尤其是在需要排序大量数据时。
// 这个函数通过反复折半查找，找出大于或等于给定数值的最高有效位的下一个较高位。
// 例如，如果给定的数值在二进制表示中最高位是第 7 位，它将返回 8。
uint32_t getHigherMsb(uint32_t n)
{
    uint32_t msb = sizeof(n) * 4; // 初始位设置为数据类型的位数的一半 (32-bit integer 有32位, 初始化为16)
    uint32_t step = msb; // 步长也初始化为位数的一半 (16)
    while (step > 1) // 当步长大于1时进入循环
    {
        step /= 2; // 每次循环步长减半
        if (n >> msb) // 检查n的msb位右移后的值是否为1
            msb += step; // 如果是，msb加上当前步长
        else
            msb -= step; // 否则，msb减去当前步长
    }
    if (n >> msb) // 最后一次检查n的msb位右移后的值是否为1
        msb++;
    return msb; // 返回msb
}

// Wrapper method to call auxiliary(辅助) coarse(粗) frustum(视椎体) containment test.
// 包装方法调用辅助粗视锥体密闭试验。
// Mark all Gaussians that pass it.
// 被markVisible调用
__global__ void checkFrustum(int P,
	const float* orig_points,
	const float* viewmatrix,
	const float* projmatrix,
	bool* present)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	float3 p_view;
	present[idx] = in_frustum(idx, orig_points, viewmatrix, projmatrix, false, p_view);
}

// Generates one key/value pair for all Gaussian / tile overlaps. 
// Run once per Gaussian (1:N mapping).
// 为所有高斯/瓦片重叠生成一个键/值对。
// 每个高斯运行一次（1：N 映射）。
__global__ void duplicateWithKeys(
	int P, // 点数量
	const float2* points_xy, // 点的投影到图像平面中的坐标
	const float* depths, // 点的深度
	const uint32_t* offsets, // 点偏移值,并行索引使用
	uint64_t* gaussian_keys_unsorted, // 排序前的键
	uint32_t* gaussian_values_unsorted, // 排序前的值
	int* radii, // 半径数组
	dim3 grid)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Generate no key/value pair for invisible Gaussians
	if (radii[idx] > 0)
	{
		// Find this Gaussian's offset in buffer for writing keys/values.
		uint32_t off = (idx == 0) ? 0 : offsets[idx - 1];
		uint2 rect_min, rect_max;

		getRect(points_xy[idx], radii[idx], rect_min, rect_max, grid);

		// For each tile that the bounding rect overlaps, emit a 
		// key/value pair. The key is |  tile ID  |      depth      |,
		// and the value is the ID of the Gaussian. Sorting the values 
		// with this key yields Gaussian IDs in a list, such that they
		// are first sorted by tile and then by depth. 
		for (int y = rect_min.y; y < rect_max.y; y++)
		{
			for (int x = rect_min.x; x < rect_max.x; x++)
			{
				//键：由 tile ID 和深度值组合而成。
				//key = y * grid.x + x：计算 tile ID。
				//key <<= 32：左移32位，为深度值腾出空间。
				//key |= *((uint32_t*)&depths[idx])：将深度值的前32位添加到键中。
				// 这样键就是高位是grid id，低位是depth，到时候按照key排序，先按grid id排序，再按depth排序
				// 这样排序的顺序正好是我们想要的，深度排序就顺便做了，通过深度排序得到对应的键值是高斯的id数
				// 到时候通过值去取到对应的高斯数据进而进行各种运算
				uint64_t key = y * grid.x + x;
				key <<= 32;
				key |= *((uint32_t*)&depths[idx]);
				gaussian_keys_unsorted[off] = key;
				gaussian_values_unsorted[off] = idx;
				off++;
			}
		}
	}
}

// Check keys to see if it is at the start/end of one tile's range in 
// the full sorted list. If yes, write start/end of this tile. 
// Run once per instanced (duplicated) Gaussian ID.
// L是被渲染的高斯总数量
__global__ void identifyTileRanges(int L, uint64_t* point_list_keys, uint2* ranges)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= L)
		return;

	// Read tile ID from key. Update start/end of tile range if at limit.
	uint64_t key = point_list_keys[idx];
	uint32_t currtile = key >> 32;
	// 给当前tile的ranges赋值
	if (idx == 0)
		ranges[currtile].x = 0;
	else
	{
		uint32_t prevtile = point_list_keys[idx - 1] >> 32;
		// 这个函数是在线程上运行的，因此线程数减一就是看前一个线程所在的格子和自己的格子是否是同一个格子
		// 如果是同一个格子，则说明当前像素和上一个像素属于同一个格子，就说明现在的像素位置小于等于边界，所以就还不能得出边界idx数
		// 只有当当前格子和之前格子不同时，说明线程跨越了一个grid，到了一个新的grid，这个时候，当前idx就是上一个格子的上界，和当前格子的下界
		// 因此这个ranges是一个做闭右开的区间
		if (currtile != prevtile)
		{
			ranges[prevtile].y = idx;
			ranges[currtile].x = idx;
		}
	}
	// 如果idx是最后一个格子，那么就说明最后一个格子的上界就是L
	if (idx == L - 1)
		ranges[currtile].y = L;
}

// Mark Gaussians as visible/invisible, based on view frustum testing(视锥体试验)
void CudaRasterizer::Rasterizer::markVisible(
	int P,
	float* means3D,
	float* viewmatrix,
	float* projmatrix,
	bool* present)
{
	checkFrustum << <(P + 255) / 256, 256 >> > (// P是点云数量，这就是开满了所有线程,一个线程计算一个check
		P,
		means3D,
		viewmatrix, projmatrix,
		present);
}

CudaRasterizer::GeometryState CudaRasterizer::GeometryState::fromChunk(char*& chunk, size_t P)
{
	GeometryState geom;
	obtain(chunk, geom.depths, P, 128);
	obtain(chunk, geom.clamped, P * 3, 128);
	obtain(chunk, geom.internal_radii, P, 128);
	obtain(chunk, geom.means2D, P, 128);
	obtain(chunk, geom.cov3D, P * 6, 128);
	obtain(chunk, geom.conic_opacity, P, 128);
	obtain(chunk, geom.rgb, P * 3, 128);
	obtain(chunk, geom.tiles_touched, P, 128);
	// 计算出来了每个高斯点覆盖的tile的数量的前缀和
	cub::DeviceScan::InclusiveSum(nullptr, geom.scan_size, geom.tiles_touched, geom.tiles_touched, P);
	obtain(chunk, geom.scanning_space, geom.scan_size, 128);
	obtain(chunk, geom.point_offsets, P, 128);
	return geom;
}

CudaRasterizer::ImageState CudaRasterizer::ImageState::fromChunk(char*& chunk, size_t N)
{
	ImageState img;
	obtain(chunk, img.accum_alpha, N, 128);
	obtain(chunk, img.n_contrib, N, 128);
	obtain(chunk, img.ranges, N, 128);
	return img;
}

CudaRasterizer::BinningState CudaRasterizer::BinningState::fromChunk(char*& chunk, size_t P)
{
	BinningState binning;
	obtain(chunk, binning.point_list, P, 128);
	obtain(chunk, binning.point_list_unsorted, P, 128);
	obtain(chunk, binning.point_list_keys, P, 128);
	obtain(chunk, binning.point_list_keys_unsorted, P, 128);
	// unsorted 就是输入项，不带unsorted就是输出项
	// binning.sorting_size 是待排序元素数量
	// nullptr 的地方本来是可以传入一个指针的，这样就可以返回这个临时存储空间的指针，但是这里我们只是计算所需空间大小，因此不需要
	// binning.sorting_size用于存储所需的临时存储空间的大小
	// binning.point_list_keys_unsorted，即未排序的键数组
	// binning.point_list_keys，即排序后的键数组
	// binning.point_list_unsorted，即未排序的值数组
	// binning.point_list，即排序后的值数组
	// P，即键值对的数量
	cub::DeviceRadixSort::SortPairs(
		nullptr, binning.sorting_size,
		binning.point_list_keys_unsorted, binning.point_list_keys,
		binning.point_list_unsorted, binning.point_list, P);
	obtain(chunk, binning.list_sorting_space, binning.sorting_size, 128);
	return binning;
}

// Forward rendering procedure for differentiable rasterization
// of Gaussians.
int CudaRasterizer::Rasterizer::forward(
	std::function<char* (size_t)> geometryBuffer,
	std::function<char* (size_t)> binningBuffer,
	std::function<char* (size_t)> imageBuffer,
	const int P, int D, int M,
	const float* background,
	const int width, int height,
	const float* means3D,
	const float* shs,
	const float* colors_precomp,
	const float* opacities,
	const float* scales,
	const float scale_modifier,
	const float* rotations,
	const float* cov3D_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const float* cam_pos,
	const float tan_fovx, float tan_fovy,
	const bool prefiltered,
	float* out_color,
	int* radii,
	bool debug)
{
	/*
	这段代码计算了相机的焦距（focal length）在图像的纵向和横向方向上的值。
	具体来说，focal_y 和 focal_x 是计算得到的焦距值，表示相机在垂直方向和水平方向上的焦距。
	tan_fovy 和 tan_fovx 是视场角的切线值，用于计算焦距。
	focal_y和focal_x不应该是一样的吗？好怪
	*/
	const float focal_y = height / (2.0f * tan_fovy);
	const float focal_x = width / (2.0f * tan_fovx);

	size_t chunk_size = required<GeometryState>(P); // 计算了GeometryState所需的大小
	char* chunkptr = geometryBuffer(chunk_size); // 这个是接受了要分配的变量的大小chunk_size，然后返回一个指向分配的内存块的指针
	GeometryState geomState = GeometryState::fromChunk(chunkptr, P); // 申请了P个点云大小的存储集合状态的变量信息的变量

	if (radii == nullptr) // 不知道是干嘛的
	{
		radii = geomState.internal_radii;
	}

	// Create Tiles
	dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);// 这个是划分成的tile的网格的维度，是以tile计数的
	dim3 block(BLOCK_X, BLOCK_Y, 1); // 描述一个块中是多少个像素乘多少个像素

	// Dynamically resize image-based auxiliary buffers during training
	// 分配图片的存储空间,在运算过程中记录结果，图片像素范围，每个像素有几个核投影，每个像素的不透明度累积值
	size_t img_chunk_size = required<ImageState>(width * height);
	char* img_chunkptr = imageBuffer(img_chunk_size);
	ImageState imgState = ImageState::fromChunk(img_chunkptr, width * height);

	if (NUM_CHANNELS != 3 && colors_precomp == nullptr)
	{
		throw std::runtime_error("For non-RGB, provide precomputed Gaussian colors!");
	}

	// Run preprocessing per-Gaussian (transformation, bounding, conversion of SHs to RGB)
	// 它将3D点云转换为2D屏幕坐标
	// 计算与渲染相关的各种属性（如深度、协方差矩阵、颜色等）
	// 函数参数列表提供了输入数据和输出缓冲区，以及一些用于计算的辅助数据
	CHECK_CUDA(FORWARD::preprocess(
		P, D, M,
		means3D,
		(glm::vec3*)scales,
		scale_modifier,
		(glm::vec4*)rotations,
		opacities,
		shs,
		geomState.clamped,
		cov3D_precomp,
		colors_precomp,
		viewmatrix, projmatrix,
		(glm::vec3*)cam_pos,
		width, height,
		focal_x, focal_y,
		tan_fovx, tan_fovy,
		radii,
		geomState.means2D,
		geomState.depths,
		geomState.cov3D,
		geomState.rgb,
		geomState.conic_opacity,
		tile_grid,
		geomState.tiles_touched,
		prefiltered
	), debug)

	// 看到这里了
	// Compute prefix sum over full list of touched tile counts by Gaussians
	// E.g., [2, 3, 0, 2, 1] -> [2, 5, 5, 7, 8]
	/*
	cub::DeviceScan::InclusiveSum：这是一个CUB库中的函数，用于在GPU上执行前缀和（scan）操作。
	前缀和是一种扫描操作，将输入数组转换为累加和数组。
	geomState.scanning_space：扫描操作所需的临时存储空间。
	geomState.scan_size：临时存储空间的大小。
	geomState.tiles_touched：输入数组，包含每个高斯点覆盖的tile数量。
	geomState.point_offsets：输出数组，包含每个高斯点覆盖的tile数量的前缀和。
	P：输入数组的大小，即高斯点的数量。
	*/
	CHECK_CUDA(cub::DeviceScan::InclusiveSum(geomState.scanning_space, geomState.scan_size, geomState.tiles_touched, geomState.point_offsets, P), debug)

	// Retrieve total number of Gaussian instances to launch and resize aux buffers
	int num_rendered;
	// geomState.point_offsets + P - 1：源内存地址，在设备（GPU）上，指向前缀和数组的最后一个元素，即总的高斯点实例数量。
	// &num_rendered：目标内存地址，在主机（CPU）上，用于存储从设备（GPU）复制过来的数据。
	// sizeof(int)：复制数据的大小，这里是一个整数的大小。
	// 从设备（GPU）上的前缀和数组的最后一个元素复制数据到主机（CPU），获取总共需要渲染的高斯点的数量。
	CHECK_CUDA(cudaMemcpy(&num_rendered, geomState.point_offsets + P - 1, sizeof(int), cudaMemcpyDeviceToHost), debug);

	// 计算排序所需的数据块大小
	size_t binning_chunk_size = required<BinningState>(num_rendered);
	// 分配数据块
	char* binning_chunkptr = binningBuffer(binning_chunk_size);
	// 从数据块中初始化 BinningState 对象
	BinningState binningState = BinningState::fromChunk(binning_chunkptr, num_rendered);

	// For each instance to be rendered, produce adequate [ tile | depth ] key 
	// and corresponding dublicated Gaussian indices to be sorted
	duplicateWithKeys << <(P + 255) / 256, 256 >> > (
		P,
		geomState.means2D,
		geomState.depths,
		geomState.point_offsets,
		binningState.point_list_keys_unsorted,
		binningState.point_list_unsorted,
		radii,
		tile_grid)
	CHECK_CUDA(, debug)

	// 这里调用 getHigherMsb 函数，计算 tile_grid.x * tile_grid.y 的最高有效位的下一个较高位。
	int bit = getHigherMsb(tile_grid.x * tile_grid.y);

	// Sort complete list of (duplicated) Gaussian indices by keys
	/*这里的参数含义如下：
	binningState.list_sorting_space：排序过程中使用的临时空间。
	binningState.sorting_size：排序所需的临时空间大小。
	binningState.point_list_keys_unsorted 和 binningState.point_list_keys：排序前后的键。
	binningState.point_list_unsorted 和 binningState.point_list：排序前后的值。
	num_rendered：要排序的键值对数量。
	0：排序起始位。
	32 + bit：排序的总位数。32表示深度的位数，bit表示 tile ID 的位数。*/
	// 这种排序方式确保了高斯点在同一个 tile 中按深度从近到远的顺序排列。
	CHECK_CUDA(cub::DeviceRadixSort::SortPairs(
		binningState.list_sorting_space,
		binningState.sorting_size,
		binningState.point_list_keys_unsorted, binningState.point_list_keys,
		binningState.point_list_unsorted, binningState.point_list,
		num_rendered, 0, 32 + bit), debug)

	/*
		_​cudaError_t cudaMemset ( void* devPtr, int  value, size_t count )

		Initializes or sets device memory to a value.
		Parameters
		devPtr
		- Pointer to device memory
		value
		- Value to set for each byte of specified memory
		count
		- Size in bytes to set
	*/
	// imgState.ranges：指向设备内存中的 uint2 类型数组，该数组存储了图像中每个 tile 的范围信息。
	// 给imgState指针指向的数据内存进行初始化
	// tile_grid.x * tile_grid.y * sizeof(uint2)表示要初始化的内存块的大小，这个小于最开始给ranges的范围，
	CHECK_CUDA(cudaMemset(imgState.ranges, 0, tile_grid.x * tile_grid.y * sizeof(uint2)), debug);

	// Identify start and end of per-tile workloads in sorted list
	// 为imgState.ranges赋值，内容是每个 tile 工作的像素范围
	if (num_rendered > 0)
		identifyTileRanges << <(num_rendered + 255) / 256, 256 >> > (
			num_rendered,
			binningState.point_list_keys,
			imgState.ranges);
	CHECK_CUDA(, debug)

	// Let each tile blend its range of Gaussians independently in parallel
	// 如果存在预计算颜色数据 (colors_precomp)，使用这些数据进行渲染。
	// 如果没有预计算颜色数据，则使用默认的 RGB 数据 (geomState.rgb) 进行渲染。
	// feature_ptr中存储着从sh中预计算出来的颜色数据,如果有事先传入的颜色数据，则使用传入的颜色数据
	// 如果没有就用从sh中计算出来的geomState.rgb
	const float* feature_ptr = colors_precomp != nullptr ? colors_precomp : geomState.rgb;
	CHECK_CUDA(FORWARD::render(
		tile_grid, block,
		imgState.ranges,
		binningState.point_list,
		width, height,
		geomState.means2D,
		feature_ptr,
		geomState.conic_opacity,
		imgState.accum_alpha,
		imgState.n_contrib,
		background,
		out_color), debug)

	return num_rendered;
}

// Produce necessary gradients for optimization, corresponding
// to forward render pass
void CudaRasterizer::Rasterizer::backward(
	const int P, int D, int M, int R,
	const float* background,
	const int width, int height,
	const float* means3D,
	const float* shs,
	const float* colors_precomp,
	const float* scales,
	const float scale_modifier,
	const float* rotations,
	const float* cov3D_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const float* campos,
	const float tan_fovx, float tan_fovy,
	const int* radii,
	char* geom_buffer,
	char* binning_buffer,
	char* img_buffer,
	const float* dL_dpix,
	float* dL_dmean2D,
	float* dL_dconic,
	float* dL_dopacity,
	float* dL_dcolor,
	float* dL_dmean3D,
	float* dL_dcov3D,
	float* dL_dsh,
	float* dL_dscale,
	float* dL_drot,
	bool debug)
{
	GeometryState geomState = GeometryState::fromChunk(geom_buffer, P);
	BinningState binningState = BinningState::fromChunk(binning_buffer, R);
	ImageState imgState = ImageState::fromChunk(img_buffer, width * height);

	if (radii == nullptr)
	{
		radii = geomState.internal_radii;
	}

	const float focal_y = height / (2.0f * tan_fovy);
	const float focal_x = width / (2.0f * tan_fovx);

	const dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	const dim3 block(BLOCK_X, BLOCK_Y, 1);

	// Compute loss gradients w.r.t. 2D mean position, conic matrix,
	// opacity and RGB of Gaussians from per-pixel loss gradients.
	// If we were given precomputed colors and not SHs, use them.
	const float* color_ptr = (colors_precomp != nullptr) ? colors_precomp : geomState.rgb;
	CHECK_CUDA(BACKWARD::render(
		tile_grid,
		block,
		imgState.ranges,
		binningState.point_list,
		width, height,
		background,
		geomState.means2D,
		geomState.conic_opacity,
		color_ptr,
		imgState.accum_alpha,
		imgState.n_contrib,
		dL_dpix,
		(float3*)dL_dmean2D,
		(float4*)dL_dconic,
		dL_dopacity,
		dL_dcolor), debug)

	// Take care of the rest of preprocessing. Was the precomputed covariance
	// given to us or a scales/rot pair? If precomputed, pass that. If not,
	// use the one we computed ourselves.
	const float* cov3D_ptr = (cov3D_precomp != nullptr) ? cov3D_precomp : geomState.cov3D;
	CHECK_CUDA(BACKWARD::preprocess(P, D, M,
		(float3*)means3D,
		radii,
		shs,
		geomState.clamped,
		(glm::vec3*)scales,
		(glm::vec4*)rotations,
		scale_modifier,
		cov3D_ptr,
		viewmatrix,
		projmatrix,
		focal_x, focal_y,
		tan_fovx, tan_fovy,
		(glm::vec3*)campos,
		(float3*)dL_dmean2D,
		dL_dconic,
		(glm::vec3*)dL_dmean3D,
		dL_dcolor,
		dL_dcov3D,
		dL_dsh,
		(glm::vec3*)dL_dscale,
		(glm::vec4*)dL_drot), debug)
}