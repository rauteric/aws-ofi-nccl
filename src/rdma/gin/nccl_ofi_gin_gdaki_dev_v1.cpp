/*
 * Copyright (c) 2026 Amazon.com, Inc. or its affiliates. All rights reserved.
 *
 * Builders for the frozen backendVersion-1 layouts declared in
 * nccl_ofi_gin_gdaki_dev_v1.h. Transcribed from
 * 81c4dc7:CUDA/host/efa_cuda_dp.cpp, with the allocation swapped from the CUDA
 * Runtime API to the plugin's accelerator abstraction.
 *
 * As with the structs: DO NOT EDIT to track upstream. Add a v2.
 */

#include "config.h"

#include <errno.h>
#include <stdint.h>
#include <string.h>

#include "efa_cuda_dp.h"
#include "nccl_ofi_cuda.h"
#include "nccl_ofi_log.h"
#include "rdma/gin/nccl_ofi_gin_gdaki_dev_v1.h"

/* Upload a host-built descriptor to a fresh GPU allocation. */
static int gdaki_upload_descriptor(void **d_out, const void *h_src, size_t len)
{
	void *d_mem = NULL;
	int ret;

	ret = nccl_net_ofi_gpu_mem_alloc(&d_mem, len);
	if (ret != 0) {
		NCCL_OFI_WARN("gin GDAKI: failed to allocate %zu bytes of GPU memory "
			      "for a queue descriptor",
			      len);
		return -ENOMEM;
	}

	/* nccl_net_ofi_gpu_mem_copy_host_to_device takes a non-const src. */
	ret = nccl_net_ofi_gpu_mem_copy_host_to_device(d_mem, const_cast<void *>(h_src), len);
	if (ret != 0) {
		NCCL_OFI_WARN("gin GDAKI: failed to copy a queue descriptor to GPU memory");
		nccl_net_ofi_gpu_mem_free(d_mem);
		return -EIO;
	}

	*d_out = d_mem;
	return 0;
}


/* Assignment order below is upstream's, so the two can be diffed side by side:
 * efa_cuda_create_qp is at efa_cuda_dp.cpp:74, efa_cuda_create_cq at :27. */

int nccl_ofi_gdaki_init_qp_v1(struct efa_cuda_qp_v1 **d_qp,
			      const struct efa_cuda_qp_attrs *attrs)
{
	if (d_qp == NULL || attrs == NULL) {
		return -EINVAL;
	}

	if (attrs->reserved != 0) {
		NCCL_OFI_WARN("gin GDAKI: qp_attrs.reserved is non-zero");
		return -EINVAL;
	}

	if (__builtin_popcount(attrs->sq_num_entries) != 1 ||
	    __builtin_popcount(attrs->rq_num_entries) != 1) {
		NCCL_OFI_WARN("gin GDAKI: SQ (%u) and RQ (%u) sizes must be positive "
			      "powers of 2",
			      attrs->sq_num_entries, attrs->rq_num_entries);
		return -EINVAL;
	}

	/* Zero-init is load-bearing: comp_mask 0 means "compatible" per
	 * efa_cuda_is_qp_compatible(), max_sge is never assigned by upstream
	 * either, and the GIN device path repurposes wqes_posted /
	 * wqes_completed as cursors that must start at zero. */
	struct efa_cuda_qp_v1 h_qp = {};

	h_qp.sq.wq.buf = attrs->sq_buffer;
	h_qp.sq.wq.db = attrs->sq_doorbell;
	h_qp.sq.wq.max_wqes = attrs->sq_num_entries;
	h_qp.sq.wq.max_batch = attrs->sq_max_batch;
	h_qp.sq.wq.queue_mask = attrs->sq_num_entries - 1;
	h_qp.sq.wq.queue_size_shift = (uint32_t)__builtin_ctz(attrs->sq_num_entries);
	h_qp.sq.wq.wqes_pending = 0;
	h_qp.sq.wq.wqes_posted = 0;
	h_qp.sq.wq.wqes_completed = 0;
	h_qp.sq.wq.pc = 0;
	h_qp.sq.wq.phase = 0;
	/* Upstream 0.0.1 hardcodes these ("TODO: get from args or delete");
	 * 0.0.2 promoted them to qp_attrs fields. Frozen at 0.0.1's values,
	 * ignoring any the caller supplies. */
	h_qp.sq.max_inline_data = 32;
	h_qp.sq.max_rdma_sges = 2;

	h_qp.rq.wq.buf = attrs->rq_buffer;
	h_qp.rq.wq.db = attrs->rq_doorbell;
	h_qp.rq.wq.max_wqes = attrs->rq_num_entries;
	h_qp.rq.wq.max_batch = attrs->rq_num_entries;
	h_qp.rq.wq.queue_mask = attrs->rq_num_entries - 1;
	h_qp.rq.wq.queue_size_shift = (uint32_t)__builtin_ctz(attrs->rq_num_entries);
	h_qp.rq.wq.wqes_pending = 0;
	h_qp.rq.wq.wqes_posted = 0;
	h_qp.rq.wq.wqes_completed = 0;
	h_qp.rq.wq.pc = 0;
	h_qp.rq.wq.phase = 1;

	void *d_mem = NULL;
	int ret = gdaki_upload_descriptor(&d_mem, &h_qp, sizeof(h_qp));
	if (ret != 0) {
		return ret;
	}
	*d_qp = static_cast<struct efa_cuda_qp_v1 *>(d_mem);
	return 0;
}

void nccl_ofi_gdaki_destroy_qp_v1(struct efa_cuda_qp_v1 *d_qp)
{
	if (d_qp != NULL) {
		nccl_net_ofi_gpu_mem_free(d_qp);
	}
}

int nccl_ofi_gdaki_init_cq_v1(struct efa_cuda_cq_v1 **d_cq,
			      const struct efa_cuda_cq_attrs *attrs)
{
	if (d_cq == NULL || attrs == NULL) {
		return -EINVAL;
	}

	if (__builtin_popcount(attrs->num_entries) != 1) {
		NCCL_OFI_WARN("gin GDAKI: CQ size (%u) must be a positive power of 2",
			      attrs->num_entries);
		return -EINVAL;
	}

	struct efa_cuda_cq_v1 h_cq = {};

	h_cq.buf = attrs->buffer;
	h_cq.entry_size = attrs->entry_size;
	h_cq.num_entries = attrs->num_entries;
	h_cq.queue_mask = attrs->num_entries - 1;
	h_cq.queue_size_shift = (uint32_t)__builtin_ctz(attrs->num_entries);
	h_cq.cc = 0;
	h_cq.phase = 1;

	void *d_mem = NULL;
	int ret = gdaki_upload_descriptor(&d_mem, &h_cq, sizeof(h_cq));
	if (ret != 0) {
		return ret;
	}
	*d_cq = static_cast<struct efa_cuda_cq_v1 *>(d_mem);
	return 0;
}

void nccl_ofi_gdaki_destroy_cq_v1(struct efa_cuda_cq_v1 *d_cq)
{
	if (d_cq != NULL) {
		nccl_net_ofi_gpu_mem_free(d_cq);
	}
}
