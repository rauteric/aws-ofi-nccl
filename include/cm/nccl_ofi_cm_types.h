/*
 * Copyright (c) 2025 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#ifndef NCCL_OFI_CM_TYPES_H_
#define NCCL_OFI_CM_TYPES_H_

#include <rdma/fabric.h>

#include "nccl_ofi.h"

/* Forward class declarations */
class nccl_ofi_connection_manager;
class nccl_ofi_cm_send_connector;
class nccl_ofi_cm_receiver_info;
class nccl_ofi_cm_listener;

/* Struct types */
typedef char ep_name[MAX_EP_ADDR];

struct cm_ep_name {
	ep_name name;
	size_t name_len;
};

struct nccl_ofi_cm_mr_handle {
	uint64_t mr_key;
	nccl_ofi_connection_manager *cm;
	fid_mr *mr;
};

struct nccl_ofi_cm_conn_msg {

	enum {
		SEND_CONN_MSG,
		SEND_CONN_RESP_MSG
	} type;

	/* A comm identitifer that uniquely identifies the comm on the local side
	   (the sender of this conn msg). The receiver must use this ID when
	   sending messages to sender */
	uint32_t local_comm_id;

	/* A comm identitifer that uniquely identifies the comm on the remote side
	   (the receiver of this conn msg) */
	uint32_t remote_comm_id;

	/* Endpoint used for connection establishment
	   listener's ep is also transmitted in the handle */
	cm_ep_name conn_ep_name;

	/* User (transport) data will be at the end of the conn msg */
};

#endif /* NCCL_OFI_CM_TYPES_H_ */
 