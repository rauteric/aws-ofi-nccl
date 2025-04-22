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

/**
 * A listener returned by connection_manager::listen(), used to accept
 * incoming connections from the listen() and create recv handles
 */
class nccl_ofi_cm_listener
{
public:
	/**
	 * Obtain the handle associated with this listener
	 *
	 * The caller (transport) should return this handle back to NCCL to
	 * be delivered out-of-band to the remote (send side) node
	 */
	nccl_net_ofi_conn_handle get_handle() { return handle; }

	/**
	 * Accept an incoming connect message from the send side to this listener
	 * This returns a receiver_info object that can be used to create the
	 * protocol-specific communicator
	 */
	nccl_ofi_cm_receiver_info *accept();

	/* --------------------------------------------------------- */
	/* Not for use by transport code */

	/**
	 * Construct a new cm listener object
	 *
	 * Not to be called by the transport code -- the transport code
	 * should use connection_manager::listen() to obtain a cm_l_comm
	 */
	nccl_ofi_cm_listener(nccl_ofi_connection_manager *cm);
	~nccl_ofi_cm_listener();

	int process_conn_msg(const nccl_ofi_cm_conn_msg &conn_msg);

	void insert_completed_r_comm(nccl_ofi_cm_receiver_info *r_comm)
	{
		pending_r_comm.push_back(r_comm);
	}
private:
	nccl_ofi_connection_manager *cm;
	uint32_t l_comm_id;
	nccl_net_ofi_conn_handle handle;
	std::deque<nccl_ofi_cm_receiver_info *> pending_r_comm;
};

/**
 * Information for the receiver side of a completed connection, intended to be
 * used by the transport to create a recv communicator. This is the object
 * returned by cm_listener->accept() when a corresponding connection is
 * established.
 */
class nccl_ofi_cm_receiver_info
{
public:
	/**
	 * Get list of rails advertised by the corresponding cm_s_comm.
	 */
	nccl_ofi_cm_ep_rail_info get_sender_ep_rails();

	/**
	 * Return opaque output pointer from the transport's rail selector
	 * for this recv connection
	 */
	void *get_rail_selector_output() {
		return nullptr;
	}

	uint32_t get_comm_id() {return r_comm_id;}

	/* --------------------------------------------------------- */
	/* Not for use by transport code */

	/**
	 * Construct a new cm_r_comm. Should not be used directly by caller;
	 * caller should obtain cm_r_comm through cm_l_comm::accept()
	 */
	nccl_ofi_cm_receiver_info(nccl_ofi_connection_manager *_cm,
			   nccl_ofi_cm_listener *_cm_l_comm,
			   const nccl_ofi_cm_conn_msg &_conn_msg);
	~nccl_ofi_cm_receiver_info();

	int send_conn_resp_msg();

	void set_conn_resp_msg_delivered() {cm_l_comm->insert_completed_r_comm(this);}

	void set_ep_rail_info(const nccl_ofi_cm_ep_rail_info &_ep_rail_info)
	{
		ep_rail_info = _ep_rail_info;
		prepare_conn_resp_msg();
	}

	fi_addr_t dest_addr;

	void *rail_selector_output;

private:
	/* Back-pointer to connection manager */
	nccl_ofi_connection_manager *cm;
	/* Back-pointer to l_comm */
	nccl_ofi_cm_listener *cm_l_comm;
	nccl_ofi_freelist_elem_t *send_elem;
	uint32_t r_comm_id;
	nccl_ofi_cm_conn_msg conn_msg;
	nccl_ofi_cm_send_conn_resp_req send_conn_resp_req;
	std::optional<nccl_ofi_cm_ep_rail_info> ep_rail_info;
	void *transport_ep;

	void prepare_conn_resp_msg();
};

/**
 * A connector returned by connection_manager::connect(), used to connect
 * to a remote node (recv side) given a handle.
 */
class nccl_ofi_cm_send_connector
{
public:
	/**
	 * Test whether the connection is complete. This will return ready=true
	 * when the connect message has been delivered and the connect response
	 * message has been received.
	 *
	 * @param ready: set to true when connection complete
	 * @return: negative errno code on error (in sending the connect response
	 * 	    message)
	 */
	int test_ready(bool *ready);

	/**
	 * Get list of rails advertised by the corresponding cm_r_comm from receiver
	 * side.
	 */
	nccl_ofi_cm_ep_rail_info get_receiver_ep_rails();

	/**
	 * Get communicator ID
	 */
	uint32_t get_comm_id() {return s_comm_id;}

	/* --------------------------------------------------------- */
	/* Not for use by transport code */

	/**
	 * Construct a new cm_s_comm
	 *
	 * Not to be called by the transport code -- the transport code
	 * should use connection_manager::connect() to obtain a cm_s_comm
	 */
	nccl_ofi_cm_send_connector(nccl_ofi_connection_manager *cm,
			   nccl_net_ofi_conn_handle *handle,
			   const nccl_ofi_cm_ep_rail_info &ep_rail_info);
	~nccl_ofi_cm_send_connector();

	fi_addr_t dest_addr;

	void set_conn_resp_msg(const nccl_ofi_cm_conn_msg &conn_resp_msg) {
		this->received_conn_resp_msg = conn_resp_msg;
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

	void prepare_conn_msg(nccl_net_ofi_conn_handle *handle, nccl_ofi_cm_conn_msg *conn_msg);
};

#endif /* NCCL_OFI_CM_H_ */
