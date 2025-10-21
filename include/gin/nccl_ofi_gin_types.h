/*
 * Copyright (c) 2025      Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#ifndef NCCL_OFI_GIN_TYPES_H
#define NCCL_OFI_GIN_TYPES_H

#include <memory>
#include <stdexcept>
#include <stdint.h>

#include "nccl_ofi_assert.h"
#include "nccl_ofi_gdrcopy.h"

/** Taken from GDRCopy header */
#define NCCL_OFI_GPU_PAGE_SHIFT   16
#define NCCL_OFI_GPU_PAGE_SIZE    (1UL << NCCL_OFI_GPU_PAGE_SHIFT)

struct nccl_ofi_gin_comm;
typedef struct nccl_ofi_gin_comm nccl_ofi_gin_comm_t;

struct nccl_net_ofi_gin_tx_req_t;
struct nccl_net_ofi_gin_iputsignal_recv_req;

struct nccl_ofi_gin_ep_t;
struct nccl_ofi_gin_ep_rail_t;

class nccl_ofi_gin_mr_handle_t;

class nccl_ofi_gin_resources;

struct nccl_net_ofi_gin_signal_metadata_msg_t {
	/* Message sequence number */
	uint32_t msg_seq_num;

	/* A comm identitifer that uniquely identifies the comm
	* on the receiver side */
	uint32_t remote_comm_id;

	uint32_t num_segments;

	uint64_t signal_base_address;
	uint64_t signal_offset;
	uint64_t signal_value;
};

struct nccl_ofi_gin_ctx {
	std::unique_ptr<nccl_ofi_gdrcopy_ctx> gdrcopy_ctx;

	nccl_ofi_gin_ctx() {
		gdrcopy_ctx.reset(nccl_ofi_gdrcopy_ctx::create());
		if (!gdrcopy_ctx) {
			throw std::runtime_error("Failed to create GDRcopy context");
		}
	}
};

/**
 * Format of immediate data:
 * 
 * | 2-bit segment count | 20-bit comm ID | 10-bit msg_seq_num |
 */
#define GIN_IMM_NUM_SEQ_BITS 10
#define GIN_IMM_COMM_BITS 20
#define GIN_MAX_COMMS (1 << GIN_IMM_COMM_BITS)
#define GIN_IMM_SEG_SHIFT (GIN_IMM_NUM_SEQ_BITS + GIN_IMM_COMM_BITS)
#define GIN_IMM_NUM_SEG_BITS 2
#define GIN_IMM_SEQ_MASK ((1 << GIN_IMM_NUM_SEQ_BITS) - 1)
#define GIN_IMM_GET_SEQ_NUM(data) ((data) & GIN_IMM_SEQ_MASK)
#define GIN_IMM_GET_COMM_ID(data) (((data) >> GIN_IMM_NUM_SEQ_BITS) & ((1 << GIN_IMM_COMM_BITS) - 1))
#define GIN_IMM_GET_SEG_CNT(data) ((data) >> GIN_IMM_SEG_SHIFT)
#define GIN_IMM_GET_IMM_DATA(comm_id, msg_seq_num, nseg) \
	(((nseg) << GIN_IMM_SEG_SHIFT) | ((comm_id) << GIN_IMM_NUM_SEQ_BITS) | (msg_seq_num))

#endif
