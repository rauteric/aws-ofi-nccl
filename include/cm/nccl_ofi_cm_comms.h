/*
 * Copyright (c) 2025 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#ifndef NCCL_OFI_CM_COMMS_H_
#define NCCL_OFI_CM_COMMS_H_

#include <deque>
#include <optional>

#include "cm/nccl_ofi_cm_types.h"
#include "cm/nccl_ofi_cm_reqs.h"
#include "nccl_ofi_freelist.h"

class nccl_ofi_cm_r_comm
{
public:
	nccl_ofi_cm_r_comm(nccl_ofi_connection_manager *cm,
			   const nccl_ofi_cm_conn_msg &conn_msg);
	~nccl_ofi_cm_r_comm();
	int test_ready(bool *ready);

	void set_ep_rail_info(const nccl_ofi_cm_ep_rail_info &_ep_rail_info)
	{
		this->ep_rail_info = _ep_rail_info;
	}

	void set_conn_resp_msg_delivered() { conn_resp_msg_delivered = true; }

	void prepare_conn_resp_msg(nccl_ofi_cm_conn_msg *conn_resp_msg);

	fi_addr_t dest_addr;
private:
	/* Back-pointer to connection manager */
	nccl_ofi_connection_manager *cm;
	nccl_ofi_freelist_elem_t *send_elem;
	uint32_t r_comm_id;
	nccl_ofi_cm_conn_msg conn_msg;
	nccl_ofi_cm_send_conn_resp_req send_conn_resp_req;
	bool conn_resp_msg_sent;
	bool conn_resp_msg_delivered;
	nccl_ofi_cm_ep_rail_info ep_rail_info;
};

class nccl_ofi_cm_l_comm
{
public:
	nccl_ofi_cm_l_comm(nccl_ofi_connection_manager *cm);
	~nccl_ofi_cm_l_comm();
	nccl_ofi_cm_handle get_handle() { return handle; }
	nccl_ofi_cm_r_comm *accept();

	void insert_conn_msg(const nccl_ofi_cm_conn_msg &conn_msg);
private:
	nccl_ofi_connection_manager *cm;
	uint32_t l_comm_id;
	nccl_ofi_cm_handle handle;
	std::deque<nccl_ofi_cm_conn_msg> pending_conn_msg;
};

class nccl_ofi_cm_s_comm
{
public:
	nccl_ofi_cm_s_comm(nccl_ofi_connection_manager *cm,
			   nccl_ofi_cm_handle *handle,
			   const nccl_ofi_cm_ep_rail_info &ep_rail_info);
	~nccl_ofi_cm_s_comm();

	int test_ready(bool *ready);
	fi_addr_t dest_addr;

	void set_conn_resp_msg(const nccl_ofi_cm_conn_msg &conn_resp_msg) {
		*(this->received_conn_resp_msg) = conn_resp_msg;
	}

	void set_conn_msg_delivered() {
		conn_msg_delivered = true;
	}

private:
	/* Back-pointer to connection manager */
	nccl_ofi_connection_manager *cm;
	nccl_ofi_freelist_elem_t *send_elem;
	nccl_ofi_cm_send_conn_req send_conn_req;
	std::optional<nccl_ofi_cm_conn_msg> received_conn_resp_msg;

	bool conn_msg_sent;
	bool conn_msg_delivered;

	uint32_t s_comm_id;

	nccl_ofi_cm_ep_rail_info ep_rail_info;

	void prepare_conn_msg(nccl_ofi_cm_handle *handle, nccl_ofi_cm_conn_msg *conn_msg);
};

#endif /* NCCL_OFI_CM_H_ */
