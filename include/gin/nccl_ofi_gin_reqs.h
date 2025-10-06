/*
 * Copyright (c) 2025      Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#ifndef NCCL_OFI_GIN_REQS_H
#define NCCL_OFI_GIN_REQS_H

#include "nccl_ofi.h"
#include "gin/nccl_ofi_gin.h"
#include "gin/nccl_ofi_gin_types.h"

/** TODO use freelist-ish thing for these... **/

struct nccl_net_ofi_gin_iputsignal_recv_req
{
	unsigned total_segments;

	unsigned num_seg_completions;

	bool metadata_received;
	nccl_net_ofi_rdma_signal_metadata_msg_t metadata;
};

struct nccl_net_ofi_gin_iputsignal_req_t {

	nccl_net_ofi_req_t base;

	uint32_t peer_rank;

	/* Associated Comm object */
	nccl_ofi_gin_comm_t *gin_comm;

	/* Message sequence number */
	uint16_t msg_seq_num;

	/* Metadata fl elem */
	nccl_ofi_freelist_elem_t *metadata_elem;

	/* Subrequests */
	/* Write request */
	nccl_net_ofi_gin_tx_req_t *write_req;
	/* Metadata send request */
	nccl_net_ofi_gin_tx_req_t *send_req;
};


static inline int gin_handle_cq_entry(nccl_net_ofi_context_t *ctx,
				      struct fi_cq_entry *cq_entry_base,
				      fi_addr_t src_addr,
				      uint16_t rail_id);


static inline int gin_handle_error_entry(nccl_net_ofi_context_t *ctx,
					    struct fid_cq *cq,
					    struct fi_cq_err_entry *err_entry,
					    uint16_t rail_id);


struct nccl_net_ofi_gin_req_t {
	nccl_net_ofi_context_t ctx;

	virtual int handle_cq_entry(nccl_net_ofi_context_t *_ctx,
				    struct fi_cq_entry *cq_entry_base,
				    fi_addr_t src_addr,
				    uint16_t rail_id) = 0;

	nccl_net_ofi_gin_req_t() : ctx({.ofi_ctx = {},
					.handle_cq_entry = gin_handle_cq_entry,
					.handle_error_entry = gin_handle_error_entry})
	{ }

	virtual ~nccl_net_ofi_gin_req_t() = default;
};


struct nccl_net_ofi_gin_tx_req_t : nccl_net_ofi_gin_req_t {

	bool done = false;

	virtual int handle_cq_entry(nccl_net_ofi_context_t *_ctx,
			    struct fi_cq_entry *cq_entry_base,
			    fi_addr_t src_addr,
			    uint16_t rail_id) {
		done = true;
		return 0;
	}

	int test(bool &done_arg) {
		done_arg = this->done;
		return 0;
	}
};

/**
 * A tx request that frees itself after completion.
 * Note: must be allocated using new()!
 */
struct nccl_net_ofi_gin_writeack_req_t : nccl_net_ofi_gin_req_t {

	nccl_ofi_gin_comm_t *gin_comm;

	virtual int handle_cq_entry(nccl_net_ofi_context_t *_ctx,
			    struct fi_cq_entry *cq_entry_base,
			    fi_addr_t src_addr,
			    uint16_t rail_id) {
		assert(gin_comm->outstanding_ack_counter > 0);
		gin_comm->outstanding_ack_counter--;

		delete this;
		return 0;
	}

	nccl_net_ofi_gin_writeack_req_t(nccl_ofi_gin_comm_t *gin_comm_arg) :
		nccl_net_ofi_gin_req_t(),
		gin_comm(gin_comm_arg)
	{ }
};

static inline int gin_handle_cq_entry(nccl_net_ofi_context_t *ctx,
				      struct fi_cq_entry *cq_entry_base,
				      fi_addr_t src_addr,
				      uint16_t rail_id)
{
	nccl_net_ofi_gin_req_t *req = cpp_container_of(ctx, &nccl_net_ofi_gin_req_t::ctx);
	return req->handle_cq_entry(ctx, cq_entry_base, src_addr, rail_id);
}

static inline int gin_handle_error_entry(nccl_net_ofi_context_t *ctx,
					    struct fid_cq *cq,
					    struct fi_cq_err_entry *err_entry,
					    uint16_t rail_id)
{
	int ret = 0;
	assert(ctx);
	nccl_net_ofi_gin_req_t *req = cpp_container_of(ctx, &nccl_net_ofi_gin_req_t::ctx);

	NCCL_OFI_WARN("Request %p completed with error. RC: %d. Error: %d (%s). Completed length: %ld",
		req, err_entry->err,
		err_entry->prov_errno,
		fi_cq_strerror(cq, err_entry->prov_errno, err_entry->err_data, NULL, 0),
		(long)err_entry->len);

	/*
	 * Libfabric error codes directly map to ISO C errno values for standard
	 * error codes up to FI_ERRNO_OFFSET, and libfabric-specific error codes
	 * beyond. nccl_net_ofi_retval_translate() will figure out
	 * how to deal with these, so it is safe to pass up the err as-is.
	 * However, any special-handling for prov_errno should be handled here.
	 */
	ret = -(err_entry->err);
	return ret;
}

#endif
