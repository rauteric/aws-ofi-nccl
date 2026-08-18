/*
 * Copyright (c) 2026 Amazon.com, Inc. or its affiliates. All rights reserved.
 *
 * GDAKI backendVersion 1: the frozen efa-dp-direct 0.0.1 device-visible queue
 * layouts, transcribed from submodule commit 81c4dc7, plus the builders that
 * produce them.
 *
 * The plugin writes these descriptors into GPU memory and an independently
 * compiled NCCL kernel dereferences them, so the byte layout is a wire format
 * and a shipped kernel cannot be recompiled to match a newer one. Hence one
 * immutable layout per backendVersion, and the plugin builds exactly the one
 * NCCL asked for.
 *
 * NOTHING IN THIS FILE MAY BE MODIFIED. The structs are upstream's with "_v1"
 * appended and nothing else changed, so that
 *   git -C 3rd-party/efa-dp-direct show 81c4dc7:CUDA/common/efa_cuda_dp_types.h
 * diffs against them as a pure rename. Reordering, merging declarations, or
 * "fixing" a type (e.g. the bare `int phase`) is an ABI change that breaks
 * every NCCL built against version 1; the builders likewise must keep writing
 * the bytes 0.0.1 wrote. Nothing here may reference a type that can change
 * underneath it, which is what keeps an efa-dp-direct bump from moving these
 * layouts. To track a new upstream release, add
 * nccl_ofi_gin_gdaki_dev_v2.{h,cpp}.
 *
 * The host-side attribute structs (efa_cuda_qp_attrs, efa_cuda_cq_attrs) are
 * deliberately NOT frozen: they never cross to NCCL, and are already
 * extensible via comp_mask + inlen.
 */

#ifndef NCCL_OFI_GIN_GDAKI_DEV_V1_H_
#define NCCL_OFI_GIN_GDAKI_DEV_V1_H_

#include <stdint.h>

/* The builders take these by pointer, so declarations suffice; that keeps this
 * header free of any efa-dp-direct include path. */
struct efa_cuda_qp_attrs;
struct efa_cuda_cq_attrs;

#ifdef __cplusplus
extern "C" {
#endif

/* The layout every NCCL with EFA-GDA support through 2.31.x was compiled
 * against. DO NOT EDIT. */

struct efa_cuda_cq_v1 {
	uint64_t comp_mask;
	uint32_t entry_size;
	uint32_t num_entries;
	uint32_t queue_mask;
	uint32_t queue_size_shift;
	uint32_t cc;
	int phase;
	uint8_t *buf;
	uint32_t *db;
};

struct efa_cuda_wq_v1 {
	uint32_t max_sge;
	uint32_t max_wqes;
	uint32_t queue_mask;
	uint32_t queue_size_shift;
	uint32_t max_batch;
	uint32_t wqes_pending;
	uint32_t wqes_posted;
	uint32_t wqes_completed;
	uint32_t pc;
	int phase;
	uint8_t *buf;
	uint32_t *db;
};

struct efa_cuda_rq_v1 {
	struct efa_cuda_wq_v1 wq;
};

struct efa_cuda_sq_v1 {
	struct efa_cuda_wq_v1 wq;
	uint32_t max_inline_data;
	uint32_t max_rdma_sges;
};

struct efa_cuda_qp_v1 {
	uint64_t comp_mask;
	struct efa_cuda_sq_v1 sq;
	struct efa_cuda_rq_v1 rq;
};


/*
 * Builders for the layouts above. Version 1 needs its own because upstream's
 * efa_cuda_create_qp() / efa_cuda_create_cq() only ever build the layout the
 * submodule is currently pinned at.
 *
 * `attrs` is the current (unfrozen) host struct; these read only the fields
 * 0.0.1 had, since a field added later has no slot in the v1 layout.
 *
 * Return 0, or a negative errno. On success *d_qp / *d_cq is a GPU-resident
 * descriptor to be released with the paired destroy function.
 */
int nccl_ofi_gdaki_init_qp_v1(struct efa_cuda_qp_v1 **d_qp,
			      const struct efa_cuda_qp_attrs *attrs);
void nccl_ofi_gdaki_destroy_qp_v1(struct efa_cuda_qp_v1 *d_qp);

int nccl_ofi_gdaki_init_cq_v1(struct efa_cuda_cq_v1 **d_cq,
			      const struct efa_cuda_cq_attrs *attrs);
void nccl_ofi_gdaki_destroy_cq_v1(struct efa_cuda_cq_v1 *d_cq);

#ifdef __cplusplus
}
#endif

#endif /* NCCL_OFI_GIN_GDAKI_DEV_V1_H_ */
