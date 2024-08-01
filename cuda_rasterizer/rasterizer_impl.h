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

#pragma once

#include <iostream>
#include <vector>
#include "rasterizer.h"
#include <cuda_runtime_api.h>

namespace CudaRasterizer
{
	/// @brief obtain 函数用于从一个大的内存块 chunk 中按字节数分配对齐的内存单元 
	/// @brief 它保证对齐并更新 chunk 指针以指向下一个未使用的内存位置。
	/// @brief 这种方法有助于提高内存分配的效率，特别是在 CUDA 编程中，
	/// @brief 通常需要进行高效的内存管理。
	/// @brief 给出一个数据类型T占据的字节数、要分配的数据有几个count， 分配一个对齐内存块的T类型数据块,
	/// @brief 并返回指向该数据块后面连续内存的地址指针
	/// @brief 这个计算方法确保了 offset 是大于等于原始地址的最近对齐地址。这样做的好处是我们可以在内存中对数据进行正确的对齐，从而优化数据访问速度和提高程序性能。
	/// @tparam T 
	/// @param chunk 当前内存块的起始地址。
	/// @param ptr  存储数据的指针
	/// @param count 要分配的元素数量
	/// @param alignment 内存对齐要求，其实就是一个数据元占据多少字节
	template <typename T>
	static void obtain(char*& chunk, T*& ptr, std::size_t count, std::size_t alignment)
	{
		// 首先，将 chunk 指针转换为 std::uintptr_t 类型，这是一个无符号整数类型，
		// 足够大以容纳指针的整数表示。这样做的目的是为了方便对指针地址进行算术运算
		// reinterpret_cast<std::uintptr_t>(chunk)就是当前内存块的起始地址，
		// alignment - 1 是一个掩码，用于找到对齐边界。假设 alignment 是 16 字节，对应的掩码是 15（即 0x0F），这会将地址对齐到 16 字节的边界。
		// ~(alignment - 1) 在二进制中将所有低于对齐边界的位清零，而保留高位。
		std::size_t offset = (reinterpret_cast<std::uintptr_t>(chunk) + alignment - 1) & ~(alignment - 1); // 计算出来了一个T数据需要的偏移量
		ptr = reinterpret_cast<T*>(offset); // 这个就是分配的一段连续的内存块的首地址指针，对齐的，大小为count * sizeof(T)
		chunk = reinterpret_cast<char*>(ptr + count); // 从ptr开始加上count个T数据，得到下一个内存块的起始地址
	}

	/// @brief 几何相关的数据
	/// @param scan_size 扫描空间的大小,通常与扫描结果的存储空间有关
	/// @param depths 深度浮点数组
	/// @param scanning_space 扫描空间的字符数组，可能用于存储临时数据
	/// @param clamped 是否被裁剪数组
	/// @param internal_radii 半径整数数组?
	/// @param means2D 2D均值float2数组
	/// @param cov3D 3D协方差浮点数组
	/// @param conic_opacity 椭圆的透明度浮点数组
	/// @param rgb RGB颜色无符号整数数组
	/// @param tiles_touched 存储触摸的瓦片数的无符号整数数组
	struct GeometryState
	{
		size_t scan_size;
		float* depths;
		char* scanning_space;
		bool* clamped;
		int* internal_radii;
		float2* means2D;
		float* cov3D;
		float4* conic_opacity;
		float* rgb;
		uint32_t* point_offsets;
		uint32_t* tiles_touched;

		static GeometryState fromChunk(char*& chunk, size_t P);
	};

	/// @brief 图像相关的数据，每个像素的点的数量，有几个点参与了贡献,累积alpha值
	/// @param ranges 存储图像每个像素的点的范围
	/// @param n_contrib 存储每个像素有n个核参与了贡献
	/// @param accum_alpha 存储了每个像素的累积alpha值
	struct ImageState
	{
		uint2* ranges; // 这个ranges是一个做闭右开的区间,表示每个tile的像素id范围
		uint32_t* n_contrib;
		float* accum_alpha;

		static ImageState fromChunk(char*& chunk, size_t N);
	};

	/// @brief 初始化与分箱相关的数据
	/// @param sorting_size 排序空间的大小
	/// @param point_list_keys_unsorted 存储未排序的点列表的排序键的无符号长整型数组
	/// @param point_list_keys 存储已排序点列表的排序键的无符号长整型数组 
	/// @param point_list_unsorted 存储未排序点列表的无符号整型数组
	/// @param point_list 存储已排序点列表的无符号整型数组
	/// @param list_sorting_space 存储排序空间的字符数组，用于排序操作的临时存储
	struct BinningState
	{
		size_t sorting_size;
		uint64_t* point_list_keys_unsorted;
		uint64_t* point_list_keys;
		uint32_t* point_list_unsorted;
		uint32_t* point_list;
		char* list_sorting_space;

		static BinningState fromChunk(char*& chunk, size_t P);
	};

	/// @brief 函数计算初始化所需的显存大小
	/// @brief 它通过调用 fromChunk 函数来确定所需的内存量，然后增加一个固定的对齐值（128 字节）以确保内存对齐和足够的空间。
	/// @tparam T 这个参数可以替换为上面的几个结构体，也可以替换为自定义的结构体，相当于用一个模板简化了函数重载声明
	/// @param P 
	/// @return 
	template<typename T> 
	size_t required(size_t P)
	{
		char* size = nullptr;
		T::fromChunk(size, P);
		//required 函数计算初始化所需的显存大小。它通过调用 fromChunk 函数来确定所需的内存量，然后增加一个固定的对齐值（128 字节）以确保内存对齐和足够的空间。
		return ((size_t)size) + 128;
	}
};