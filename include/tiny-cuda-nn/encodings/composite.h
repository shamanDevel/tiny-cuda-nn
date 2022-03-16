/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without modification, are permitted
 * provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright notice, this list of
 *       conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright notice, this list of
 *       conditions and the following disclaimer in the documentation and/or other materials
 *       provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the names of its contributors may be used
 *       to endorse or promote products derived from this software without specific prior written
 *       permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TOR (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *//*
 */

/** @file   composite.h
 *  @author Thomas Müller, NVIDIA
 *  @brief  The composite encoding allows applying different, nested encodings
 *          to different dimensions of the input.
 */

#pragma once

#include "../common.h"
#include "../encoding.h"
#include "../gpu_memory.h"
#include "../common_device.h"
#include "../multi_stream.h"

#include <numeric>
#include <stdexcept>
#include <stdint.h>
#include <string>
#include <vector>

TCNN_NAMESPACE_BEGIN

template <typename T>
class CompositeEncoding : public Encoding<T> {
public:
	CompositeEncoding(const json& params, uint32_t n_dims_to_encode)
	: m_n_dims_to_encode{n_dims_to_encode} {
		if (!params.contains("nested") || !params["nested"].is_array()) {
			throw std::runtime_error{"Must provide an array of nested encodings to CompositeEncoding."};
		}

		const json::array_t& nested = params["nested"];

		int total_nested_dims_to_encode = 0;
		for (size_t i = 0; i < nested.size(); ++i) {
			if (nested[i].contains("n_dims_to_encode")) {
				auto v = nested[i]["n_dims_to_encode"];
				if (v.is_array())
				{
					if (v.size() != 2)
						throw std::runtime_error{ "n_dims_to_encode specified as array, indicating [min,max], but a different number than 2 entries are given" };
					int min = v[0].get<int>();
					int max = v[1].get<int>();
					if (min < 0) throw std::runtime_error{ "Illegal slice, min must be >= 0" };
					if (max <= min) throw std::runtime_error{ "Illegal slice, max > min required" };
					total_nested_dims_to_encode = std::max(total_nested_dims_to_encode, max);
				} else
				{
					total_nested_dims_to_encode += v.get<int>();
				}
			}
		}

		if (total_nested_dims_to_encode > n_dims_to_encode) {
			throw std::runtime_error{"CompositeEncoding:' nested encodings must not encode more dims than composite"};
		}

		uint32_t unspecified_dims_to_encode = n_dims_to_encode - total_nested_dims_to_encode;

		// Create encodings with somewhat arbitrary alignment
		int encode_start = 0;
		for (size_t i = 0; i < nested.size(); ++i) {
			int encode_min, encode_max;
			if (nested[i].contains("n_dims_to_encode")) {
				auto v = nested[i]["n_dims_to_encode"];
				if (v.is_array())
				{
					encode_min = v[0].get<int>();
					encode_max = v[1].get<int>();
				} else
				{
					encode_min = encode_start;
					encode_max = encode_start + v.get<int>();
				}
			} else {
				if (unspecified_dims_to_encode == 0xFFFFFFFF) {
					throw std::runtime_error{"CompositeEncoding: may only leave 'n_dims_to_encode' unspecified for a single nested encoding"};
				}
				encode_min = encode_start;
				encode_max = encode_start + unspecified_dims_to_encode;
				unspecified_dims_to_encode = 0xFFFFFFFF;
			}
			encode_start = encode_max;

			auto nested_dims_to_encode = encode_max - encode_min;
			if (nested_dims_to_encode > 0) {
				m_nested.push_back(Nested{
					std::unique_ptr<Encoding<T>>(create_encoding<T>(nested_dims_to_encode, nested[i], 1)),
					encode_min, nested_dims_to_encode
					});
			}
		}

		// Fix alignment such that min_alignment of each individual encoding's output is ensured
		uint32_t dims_encoded_so_far = 0;
		for (size_t i = 0; i < m_nested.size()-1; ++i) {
			uint32_t desired_alignment = m_nested[i+1].nested->min_alignment();
			uint32_t effective_alignmen_needed = next_multiple(dims_encoded_so_far, desired_alignment) - dims_encoded_so_far;

			if (effective_alignmen_needed > 0) {
				m_nested[i].nested->set_alignment(effective_alignmen_needed);
			}

			dims_encoded_so_far += m_nested[i].nested->padded_output_width();
		}
	}

	std::unique_ptr<Context> forward(
		cudaStream_t stream,
		const GPUMatrixDynamic<float>& input,
		GPUMatrixDynamic<T>* output = nullptr,
		bool use_inference_params = false,
		bool prepare_input_gradients = false
	) override {
		if (m_n_dims_to_encode == 0) {
			return std::make_unique<ForwardContext>();
		}

		auto forward = std::make_unique<ForwardContext>();
		forward->nested.resize(m_nested.size());

		SyncedMultiStream synced_streams{stream, m_nested.size()};

		//uint32_t input_offset = 0;
		uint32_t output_offset = 0;

		for (size_t i = 0; i < m_nested.size(); ++i) {
			const auto& nested = m_nested[i];
			//uint32_t input_width = nested->input_width();
			uint32_t output_width = nested.nested->output_width();

			GPUMatrixDynamic<T> sliced_output;
			if (output) {
				sliced_output = output->slice_rows(output_offset, output_width);
			}

			forward->nested[i] = nested.nested->forward(
				stream,
				input.slice_rows(nested.dim_start, nested.num_dim), //input.slice_rows(input_offset, input_width),
				output ? &sliced_output : nullptr,
				use_inference_params,
				prepare_input_gradients
			);

			//input_offset += input_width;
			output_offset += output_width;
		}

		return forward;
	}

	void backward(
		cudaStream_t stream,
		const Context& ctx,
		const GPUMatrixDynamic<float>& input,
		const GPUMatrixDynamic<T>& output,
		const GPUMatrixDynamic<T>& dL_doutput,
		GPUMatrixDynamic<float>* dL_dinput = nullptr,
		bool use_inference_params = false,
		EGradientMode param_gradients_mode = EGradientMode::Overwrite
	) override {
		if (m_n_dims_to_encode == 0) {
			return;
		}

		const auto& forward = dynamic_cast<const ForwardContext&>(ctx);
		if (forward.nested.size() != m_nested.size()) {
			throw std::runtime_error{"CompositeEncoding::backward called with incompatible context size."};
		}

		SyncedMultiStream synced_streams{stream, m_nested.size()};

		//uint32_t input_offset = 0;
		uint32_t output_offset = 0;

		for (size_t i = 0; i < m_nested.size(); ++i) {
			const auto& nested = m_nested[i];
			//uint32_t input_width = nested->input_width();
			uint32_t output_width = nested.nested->output_width();

			GPUMatrixDynamic<float> sliced_dL_dinput;
			if (dL_dinput) {
				sliced_dL_dinput = dL_dinput->slice_rows(nested.dim_start, nested.num_dim); //dL_dinput->slice_rows(input_offset, input_width);
			}

			nested.nested->backward(
				synced_streams.get(i),
				*forward.nested[i],
				input.slice_rows(nested.dim_start, nested.num_dim), //input.slice_rows(input_offset, input_width),
				output.slice_rows(output_offset, output_width),
				dL_doutput.slice_rows(output_offset, output_width),
				dL_dinput ? &sliced_dL_dinput : nullptr,
				use_inference_params,
				param_gradients_mode
			);

			//input_offset += input_width;
			output_offset += output_width;
		}
	}

	uint32_t input_width() const override {
		return m_n_dims_to_encode;
	}

	uint32_t padded_output_width() const override {
		uint32_t total = 0;
		for (const auto& nested : m_nested) {
			total += nested.nested->padded_output_width();
		}
		return total;
	}

	uint32_t output_width() const override {
		uint32_t total = 0;
		for (const auto& nested : m_nested) {
			total += nested.nested->output_width();
		}
		return total;
	}

	uint32_t required_input_alignment() const override {
		return 1;
	}

	void set_alignment(uint32_t alignment) override {
		uint32_t n_dims = padded_output_width();
		uint32_t last_n_dims = m_nested.back().nested->padded_output_width();

		uint32_t desired_n_dims = next_multiple(n_dims, alignment);
		m_nested.back().nested->set_alignment(desired_n_dims - (n_dims - last_n_dims));
	}

	uint32_t min_alignment() const override {
		return 1;
	}

	bool supports_output_layout(MatrixLayout layout) const {
		// Only supports layout if all nested encodings do
		bool result = true;
		for (const auto& nested : m_nested) {
			result &= nested.nested->supports_output_layout(layout);
		}

		return result;
	}

	MatrixLayout preferred_output_layout() const override {
		// All encodings support AoS, so if any prefers AoS, use that.
		for (const auto& nested : m_nested) {
			if (nested.nested->preferred_output_layout() == AoS) {
				return AoS;
			}
		}

		return SoA;
	}

	void initialize_params(pcg32& rnd, float* params_full_precision, T* params, T* inference_params, T* backward_params, T* gradients, float scale = 1) override {
		size_t offset = 0;
		for (auto& nested : m_nested) {
			nested.nested->initialize_params(
				rnd,
				params_full_precision + offset,
				params + offset,
				inference_params + offset,
				backward_params + offset,
				gradients + offset,
				scale
			);
			offset += nested.nested->n_params();
		}
	}

	size_t n_params() const override {
		size_t total = 0;
		for (const auto& nested : m_nested) {
			total += nested.nested->n_params();
		}
		return total;
	}

	json hyperparams() const override {
		json::array_t nested;
		for (auto& n : m_nested) {
			nested.emplace_back(n.nested->hyperparams());
			//TODO: write n_dims_to_encode into Json
		}

		return {
			{"otype", "Composite"},
			{"nested", nested}
		};
	}

private:
	struct ForwardContext : public Context {
		std::vector<std::unique_ptr<Context>> nested;
	};
	struct Nested
	{
		std::unique_ptr<Encoding<T>> nested;
		int dim_start;
		int num_dim;
	};

	std::vector<Nested> m_nested;
	uint32_t m_n_dims_to_encode;

	MatrixLayout m_output_layout = AoS;
};

TCNN_NAMESPACE_END
