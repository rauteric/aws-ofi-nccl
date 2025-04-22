/*
 * Copyright (c) 2025 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#ifndef NCCL_OFI_CM_H_
#define NCCL_OFI_CM_H_

#include <rdma/fabric.h>

#include <deque>
#include <memory>
#include <optional>
#include <unordered_map>
#include <vector>

#include "nccl_ofi.h"
#include "nccl_ofi_freelist.h"
#include "nccl_ofi_idpool.h"

#include "cm/nccl_ofi_cm_types.h"
#include "cm/nccl_ofi_cm_comms.h"

/**
 * Simple helper FunctionObject to finalize a freelist, making it usable
 * with unique_ptr.
 */
class freelist_deleter
{
public:
	void operator()(nccl_ofi_freelist_t *freelist)
	{
		nccl_ofi_freelist_fini(freelist);
	}
};

/**
 * Callback for transport to populate the connect response message
 *
 * opaque_input: provides transport-chosen state to this function
 *
 * opaque_output: state returned by this function that will be available
 * 		  in the receiver_info object once the connection is
 * 		  established
 *
 * 		  For example, this might be a pointer to the transport
 * 		  endpoint created by this function.
 *
 * TODO: The C++ way to do this, and avoid void ptr casting, is to make this
 *       a template parameter. However, the way the CM code is currently
 *       organized, this means making every other CM class a template class
 *       as well, and generally makes the code messy. May revisit this in the
 *       future
 */
typedef void (*transport_select_conn_resp_fn)(const void *transport_connect_msg,
					      void *opaque_input,
					      void *transport_connect_resp_msg,
					      void **opaque_output);

/**
 * Connection manager. Top-level class that the caller can call connection
 * establishment functions, listen(), connect, accept(), to establish
 * send/recv communicators (TODO not really communicators, rename).
 *
 * Intent is for client code to store and initialize a connection manager
 * per (transport-specific) endpoint.
 *
 * Maintains a separate fid_ep and state that will be shared across all comms
 * created using this connection manager instance. The created endpoint will
 * be bound to the caller-supplied completion queue.
 */
class nccl_ofi_connection_manager
{
public:
	/**
	 * Initializes the CM system state. Creates an endpoint and posts
	 * initial buffer pool
	 *
	 * @param info, domain: Libfabric info and domain objects against
	 * 			which the CM endpoint will be created
	 *
	 * @param cq: the completion queue to bind the new endpoint to.
	 *            Ops submitted through the CM code will have a context
	 *            pointer to nccl_net_ofi_context_t, with appropriate
	 *            completion handling functions
	 *
	 * @param user_conn_msg_size: size of transport-specific part of
	 * 			      connect and connect response messages
	 */
	nccl_ofi_connection_manager(fi_info *info, fid_domain *_domain,
				    fid_cq *cq, size_t num_comm_ids,
				    nccl_ofi_idpool_t *mr_key_pool,
				    size_t user_conn_msg_size,
				    transport_select_conn_resp_fn transport_select_conn_resp_callback,
				    void *transport_select_conn_resp_input);

	/**
	 * Destructor. Finalizes CM endpoint and other state.
	 *
	 * Note: the connection manager should not be destroyed until all
	 * associated cm communicators (TODO: rename from communicators) are
	 * destroyed
	 */
	~nccl_ofi_connection_manager();

	/**
	 * Return a communicator (TODO rename to handle or some such) that can
	 * be used to accept incoming connections from connect()
	 */
	nccl_ofi_cm_listener *listen();

	/**
	 * Given a handle (produced by cm_l_comm on a remote node), establish
	 * a (cm_s_comm) connection to the remote node. This will return a
	 * cm_s_comm handle that can be queried until the connection is fully
	 * established.
	 *
	 * @param handle: handle from cm_l_comm on a remote node
	 * @param rail_info: rail information (addresses of data-transfer endpoints to send
	 * 		     to the remote node)
	 */
	nccl_ofi_cm_send_connector *connect(nccl_net_ofi_conn_handle *handle,
					    const void *transport_connect_msg);


	/* --------------------------------------------------------- */

	/* TODO: the functions below are not intended to be used by the caller
	   (i.e., transport), but are public due to use by other classes in the
	   cm code. Revisit the separation of concerns here. */

	/**
	 * Map from comm id to l_comm. Used for receiving connect message
	 */
	nccl_ofi_cm_listener *get_l_comm(uint32_t l_comm_id);
	/**
	 * Map from comm id to s_comm. Used for receiving connect response message
	 */
	nccl_ofi_cm_send_connector *get_s_comm(uint32_t s_comm_id);

	/**
	 * Access functions for the conn_msg freelist (which takes care of memory
	 * registration)
	 */
	nccl_ofi_freelist_elem_t *alloc_conn_msg()
	{
		return nccl_ofi_freelist_entry_alloc(conn_msg_fl.get());
	}
	void free_conn_msg(nccl_ofi_freelist_elem_t *conn_msg)
	{
		nccl_ofi_freelist_entry_free(conn_msg_fl.get(), conn_msg);
	}

	/**
	 * Accessors for OFI resources
	 */
	fid_ep *get_ep() {return ep;}
	fid_domain *get_domain() {return domain;}

	/**
	 * Accessor for CM-managed address vector. Used by cm_s_comm and cm_l_comm
	 * to insert address from received connect/connect-response messages
	 *
	 * TODO just make an accessor for fid_av* instead?
	 */
	int av_insert_address(const ep_name address, fi_addr_t *fi_addr);

	const cm_ep_name &get_conn_ep_name() {return conn_ep_name;}

	nccl_ofi_idpool_t *get_l_comm_id_pool() { return &l_comm_id_pool; }
	nccl_ofi_idpool_t *get_data_comm_id_pool() { return &data_comm_id_pool; }

	std::unordered_map<uint32_t, nccl_ofi_cm_listener *> *get_l_comm_map()
	{ return &l_comm_map; }

	std::unordered_map<uint32_t, nccl_ofi_cm_send_connector *> *get_s_comm_map()
	{ return &s_comm_map; }

	nccl_ofi_idpool_t *get_mr_key_pool() {return mr_key_pool;}

	void append_pending_req(nccl_ofi_cm_req *req)
	{
		pending_reqs.push_back(req);
	}

	/**
	 * Post any reqs pending due to EAGAIN.
	 */
	int process_pending_reqs();

	void transport_select_conn_resp(const void *transport_connect_msg,
					void *transport_connect_resp_msg,
					void **opaque_output)
	{
		return transport_select_conn_resp_callback
			(transport_connect_msg, transport_select_conn_resp_input,
			 transport_connect_resp_msg, opaque_output);
	}
private:
	/* Input */
	fid_domain *domain;
	/* Created by CM */
	fid_ep *ep;
	fid_av *av;

	std::unordered_map<uint32_t, nccl_ofi_cm_listener *> l_comm_map;
	std::unordered_map<uint32_t, nccl_ofi_cm_send_connector *> s_comm_map;

	std::unique_ptr<nccl_ofi_freelist_t, freelist_deleter> conn_msg_fl;

	/* This must appear after conn_msg_fl so it is destructed first,
	   since rx_req destructor returns buffer to freelist */
	std::vector<std::unique_ptr<nccl_ofi_cm_rx_req>> rx_req_list;

	/* rx reqs that need to be posted (due to EAGAIN) */
	std::deque<nccl_ofi_cm_req *> pending_reqs;

	nccl_ofi_idpool_t l_comm_id_pool;
	nccl_ofi_idpool_t data_comm_id_pool;

	/* Input to constructor */
	nccl_ofi_idpool_t *mr_key_pool;

	cm_ep_name conn_ep_name;

	transport_select_conn_resp_fn transport_select_conn_resp_callback;
	void *transport_select_conn_resp_input;

	size_t conn_msg_size;

	void set_conn_ep_name();

	void init_rx_buffers();
};

#endif /* NCCL_OFI_CM_H_ */
