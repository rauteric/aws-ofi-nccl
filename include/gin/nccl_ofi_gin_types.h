/*
 * Copyright (c) 2025      Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#ifndef NCCL_OFI_GIN_TYPES_H
#define NCCL_OFI_GIN_TYPES_H

#include <gdrapi.h>
#include <stdint.h>

#include "nccl_ofi_assert.h"

struct nccl_ofi_gin_comm;
typedef struct nccl_ofi_gin_comm nccl_ofi_gin_comm_t;

struct nccl_net_ofi_gin_tx_req_t;
struct nccl_net_ofi_gin_iputsignal_recv_req;

struct nccl_ofi_gin_ep_t;
struct nccl_ofi_gin_ep_rail_t;

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
	gdr_t gdr_handle;

	nccl_ofi_gin_ctx() {
		gdr_handle = gdr_open();
		assert_always(gdr_handle != nullptr && "Failed to open GDR handle");
	}

	~nccl_ofi_gin_ctx() {
		int ret = gdr_close(gdr_handle);
		assert_always(ret == 0 && "Failed to close GDR handle");
	}
};

#endif
