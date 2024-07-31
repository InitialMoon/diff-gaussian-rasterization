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

#ifndef CUDA_RASTERIZER_AUXILIARY_H_INCLUDED
#define CUDA_RASTERIZER_AUXILIARY_H_INCLUDED

#include "config.h"
#include "stdio.h"

#define BLOCK_SIZE (BLOCK_X * BLOCK_Y)
#define NUM_WARPS (BLOCK_SIZE/32)

// Spherical harmonics coefficients
__device__ const float SH_C0 = 0.28209479177387814f;
__device__ const float SH_C1 = 0.4886025119029199f;
__device__ const float SH_C2[] = {
	1.0925484305920792f,
	-1.0925484305920792f,
	0.31539156525252005f,
	-1.0925484305920792f,
	0.5462742152960396f
};
__device__ const float SH_C3[] = {
	-0.5900435899266435f,
	2.890611442640554f,
	-0.4570457994644658f,
	0.3731763325901154f,
	-0.4570457994644658f,
	1.445305721320277f,
	-0.5900435899266435f
};

__forceinline__ __device__ float ndc2Pix(float v, int S)
{
	return ((v + 1.0) * S - 1.0) * 0.5;
}

/// @brief min(grid.y, xxx)保证rect_min.y在grid.y范围内
/// @brief max(0, (p.y - max_radius) / BLOCK_Y))保证xxx不要小于0,保证外层和0取最小不要变成负数
/// @brief 核心在(p.y - max_radius) / BLOCK_Y,大多数情况下这个就是最终的取值,p.y - max_radius是2d椭球下边缘包围盒的y值
/// @brief 下面的p.y + max_radius是椭球上边缘包围盒的y值
/// @brief 假设p.y = 0.5, max_radius = 0.5, BLOCK_Y = 1
/// @brief (p.y - max_radius) / BLOCK_Y = 0
/// @brief (p.y + max_radius) / BLOCK_Y = 1
/// @brief 这个就是算出来了在y轴上高斯椭球的最大和最小BLOCK_Y格子数
/// @brief 但是这有一个问题，就是如果max_radius = 0.4那么最大和最小都是0了
/// @brief BLOCK_Y和BLOCK_X其实就是一个格子中的最小刻度，
/// @brief 但是这里又个问题就是p.y不是像素为单位的吗，BLOCK值和像素又没有一一对应关系，在下面的最大值求解过程中加是要干嘛
/// @brief 而且就算加了在我的这个数值设计下，最大值还是0，因为(0.9 + 1 - 1) / 1 = 0
/// @brief 那这种情况下这个高斯核是有半径的，但是最终在外面返回后计算的结果就会是0，就直接返回了？
/// @brief 那就不需要预计算了？
/// @brief 回答疑问：这里出现这种错误是因为我们貌似取到了边界值，也就是当BLOCK_Y = 1时
/// @brief 当BLOCK_Y != 1时，下面的算法让同一个数在除以BLOCK_Y时最大是向上取整，最小是向下取整，
/// @brief 从而形成只要半径不为0，都能计算至少1*1个格子的跨度，并且还能标定是在第几个格子，最小值的下标索引
/// @brief 但是这是否有问题，为什么是用BLOCK_Y来做分母和取整的基准，而不是每个tile的像素长宽来表示
/// @param p 投影到2d平面后点的位置
/// @param max_radius 最大半径，其实就是椭球的长边 
/// @param rect_min 矩形最小值包含x和y两个分量的值
/// @param rect_max 矩形最大值包含x和y两个分量的值
/// @param grid 整个画面被分割成多少个格子,这个是格子的一个3维数组
/// @return 从rect_min和rect_max两个引用中返回
__forceinline__ __device__ void getRect(const float2 p, int max_radius, uint2& rect_min, uint2& rect_max, dim3 grid)
{
	rect_min = {
		min(grid.x, max((int)0, (int)((p.x - max_radius) / BLOCK_X))),
		min(grid.y, max((int)0, (int)((p.y - max_radius) / BLOCK_Y)))
	};
	rect_max = {
		min(grid.x, max((int)0, (int)((p.x + max_radius + BLOCK_X - 1) / BLOCK_X))),
		min(grid.y, max((int)0, (int)((p.y + max_radius + BLOCK_Y - 1) / BLOCK_Y)))
	};
	
	// // 这个才是我理解的正确的实现，因为BLOCK_Y和BLOCK_X是用来分割
	// rect_min = {
	// 	min(BLOCK_X, max((int)0, (int)((p.x - max_radius) / grid.x))),
	// 	min(BLOCK_Y, max((int)0, (int)((p.y - max_radius) / grid.y)))
	// };
	// rect_max = {
	// 	min(BLOCK_X, max((int)0, (int)((p.x + max_radius + grid.x - 1) / grid.x))),
	// 	min(BLOCK_Y, max((int)0, (int)((p.y + max_radius + grid.y - 1) / grid.y)))
	// };
}

__forceinline__ __device__ float3 transformPoint4x3(const float3& p, const float* matrix)
{
	float3 transformed = {
		matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z + matrix[12],
		matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z + matrix[13],
		matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z + matrix[14],
	};
	return transformed;
}

__forceinline__ __device__ float4 transformPoint4x4(const float3& p, const float* matrix)
{
	float4 transformed = {
		matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z + matrix[12],
		matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z + matrix[13],
		matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z + matrix[14],
		matrix[3] * p.x + matrix[7] * p.y + matrix[11] * p.z + matrix[15]
	};
	return transformed;
}

__forceinline__ __device__ float3 transformVec4x3(const float3& p, const float* matrix)
{
	float3 transformed = {
		matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z,
		matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z,
		matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z,
	};
	return transformed;
}

__forceinline__ __device__ float3 transformVec4x3Transpose(const float3& p, const float* matrix)
{
	float3 transformed = {
		matrix[0] * p.x + matrix[1] * p.y + matrix[2] * p.z,
		matrix[4] * p.x + matrix[5] * p.y + matrix[6] * p.z,
		matrix[8] * p.x + matrix[9] * p.y + matrix[10] * p.z,
	};
	return transformed;
}

__forceinline__ __device__ float dnormvdz(float3 v, float3 dv)
{
	float sum2 = v.x * v.x + v.y * v.y + v.z * v.z;
	float invsum32 = 1.0f / sqrt(sum2 * sum2 * sum2);
	float dnormvdz = (-v.x * v.z * dv.x - v.y * v.z * dv.y + (sum2 - v.z * v.z) * dv.z) * invsum32;
	return dnormvdz;
}

__forceinline__ __device__ float3 dnormvdv(float3 v, float3 dv)
{
	float sum2 = v.x * v.x + v.y * v.y + v.z * v.z;
	float invsum32 = 1.0f / sqrt(sum2 * sum2 * sum2);

	float3 dnormvdv;
	dnormvdv.x = ((+sum2 - v.x * v.x) * dv.x - v.y * v.x * dv.y - v.z * v.x * dv.z) * invsum32;
	dnormvdv.y = (-v.x * v.y * dv.x + (sum2 - v.y * v.y) * dv.y - v.z * v.y * dv.z) * invsum32;
	dnormvdv.z = (-v.x * v.z * dv.x - v.y * v.z * dv.y + (sum2 - v.z * v.z) * dv.z) * invsum32;
	return dnormvdv;
}

__forceinline__ __device__ float4 dnormvdv(float4 v, float4 dv)
{
	float sum2 = v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
	float invsum32 = 1.0f / sqrt(sum2 * sum2 * sum2);

	float4 vdv = { v.x * dv.x, v.y * dv.y, v.z * dv.z, v.w * dv.w };
	float vdv_sum = vdv.x + vdv.y + vdv.z + vdv.w;
	float4 dnormvdv;
	dnormvdv.x = ((sum2 - v.x * v.x) * dv.x - v.x * (vdv_sum - vdv.x)) * invsum32;
	dnormvdv.y = ((sum2 - v.y * v.y) * dv.y - v.y * (vdv_sum - vdv.y)) * invsum32;
	dnormvdv.z = ((sum2 - v.z * v.z) * dv.z - v.z * (vdv_sum - vdv.z)) * invsum32;
	dnormvdv.w = ((sum2 - v.w * v.w) * dv.w - v.w * (vdv_sum - vdv.w)) * invsum32;
	return dnormvdv;
}

__forceinline__ __device__ float sigmoid(float x)
{
	return 1.0f / (1.0f + expf(-x));
}

/// @brief 检查点是否在视锥体内,同时将点进行投影到NDC空间存储在p_view中
/// @param idx 
/// @param orig_points 
/// @param viewmatrix 
/// @param projmatrix 
/// @param prefiltered 预滤波
/// @param p_view 存储点在NDC空间中的坐标
/// @return bool
__forceinline__ __device__ bool in_frustum(int idx,
	const float* orig_points,
	const float* viewmatrix,
	const float* projmatrix,
	bool prefiltered,
	float3& p_view)
{
	float3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };

	// Bring points to screen space
	// float4 p_hom = transformPoint4x4(p_orig, projmatrix);
	// float p_w = 1.0f / (p_hom.w + 0.0000001f);
	// float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };
	p_view = transformPoint4x3(p_orig, viewmatrix);

	if (p_view.z <= 0.2f)// || ((p_proj.x < -1.3 || p_proj.x > 1.3 || p_proj.y < -1.3 || p_proj.y > 1.3)))
	{
		if (prefiltered)
		{
			printf("Point is filtered although prefiltered is set. This shouldn't happen!");
			__trap();
		}
		return false;
	}
	return true;
}

#define CHECK_CUDA(A, debug) \
A; if(debug) { \
auto ret = cudaDeviceSynchronize(); \
if (ret != cudaSuccess) { \
std::cerr << "\n[CUDA ERROR] in " << __FILE__ << "\nLine " << __LINE__ << ": " << cudaGetErrorString(ret); \
throw std::runtime_error(cudaGetErrorString(ret)); \
} \
}

#endif