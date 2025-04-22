/*
 * Copyright (c) 2025 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#ifndef NCCL_OFI_CM_H_
#define NCCL_OFI_CM_H_

#include <rdma/fabric.h>

#include <deque>
#include <memory>
#include <vector>

#include "nccl_ofi.h"
#include "nccl_ofi_idpool.h"

#include "cm/nccl_ofi_cm_types.h"
#include "cm/nccl_ofi_cm_comms.h"


/**
 * Callback for transport to populate the connect response message
 *
 * @param transport_connect_msg:
 * 	connect message data provided by the sender
 *
 * @param opaque_input:
 * 	provides transport-chosen state to this function. This will be equal to the pointer
 * 	passed to listener() constructor.
 *
 * @param opaque_output:
 *      state returned by this function that will be available in the
 *      receiver_info object once the connection is established
 *
 *      For example, this might be a pointer to the transport endpoint created
 *      by this function.
 */
typedef void (*transport_select_conn_resp_fn)(const void *transport_connect_msg,
					      void *opaque_input,
					      void *transport_connect_resp_msg,
					      void **opaque_output);

/**
 * A listener returned by connection_manager::listen(), used to accept incoming
 * connections from listen() and create a receiver info object
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
	 * Accept an incoming connection from the send side to this listener
	 * This returns a receiver_info object that can be used to create the
	 * protocol-specific communicator.
	 *
	 * If the connection is not yet established, returns empty.
	 *
	 * See documentation below on nccl_ofi_cm_receiver_info
	 */
	std::optional<nccl_ofi_cm_receiver_info> accept();

	/* Destructor, frees associated resources */
	~nccl_ofi_cm_listener();

private:
	/**
	 * Construct a new cm listener object.
	 *
	 * (Note: transport should use connection_manager::connect() instead
	 *  of constructing this object directly)
	 *
	 * @param cm:
	 *      An instance of the connect manager for this domain
	 *
	 * @param transport_select_conn_resp_callback:
	 *      Transport callback to create conn response message
	 *
	 * @param transport_select_conn_resp_input:
	 * 	Transport-selected state to be provided to the callback
	 */
	nccl_ofi_cm_listener(nccl_ofi_connection_manager &cm,
		transport_select_conn_resp_fn transport_select_conn_resp_callback,
		void *transport_select_conn_resp_input);

	nccl_ofi_connection_manager &cm;
	uint32_t listener_id;
	nccl_net_ofi_conn_handle handle;
	std::deque<nccl_ofi_cm_receiver_info> receiver_info_queue;

	transport_select_conn_resp_fn transport_select_conn_resp_callback;
	void *transport_select_conn_resp_input;

	friend class nccl_ofi_connection_manager;
};


/**
 * Represents a completed connection. Provides information for the receiver side
 * of a completed connection, intended to be used by the transport to create a
 * recv communicator. This is the object returned by cm_listener->accept() when
 * a corresponding connection is established.
 */
class nccl_ofi_cm_receiver_info
{
public:
	/**
	 * Return opaque output pointer provided by transport from
	 * transport_select_conn_resp_fn for this connection
	 */
	void *get_transport_select_output()
	{
		return transport_select_output;
	}

	/**
	 * Return connect message data from sender.
	 *
	 * Note: Returns a pointer to memory owned by this object. The memory
	 * is valid until this object is destroyed.
	 */
	const void *get_conn_msg()
	{
		return user_conn_msg.data();
	}

	/**
	 * Return connect response message data from sender.
	 *
	 * Note: Returns a pointer to memory owned by this object. The memory
	 * is valid until this object is destroyed.
	 */
	const void *get_conn_resp_msg()
	{
		return user_conn_resp_msg.data();
	}

private:

	nccl_ofi_cm_receiver_info(std::vector<uint8_t> conn_msg, std::vector<uint8_t> conn_resp_msg,
				  void *transport_select_output);


	std::vector<uint8_t> user_conn_msg;
	std::vector<uint8_t> user_conn_resp_msg;

	void *transport_select_output;
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
	 * Pointer to data returned from receiver side in the connect response
	 * message, once connection is complete. Note: this data is invalidated
	 * when send_connector is destroyed
	 */
	void *get_conn_resp_msg();

	/* Destructor, to free associated resources */
	~nccl_ofi_cm_send_connector();

private:

	/**
	 * Construct a new send connector
	 *
	 * (Note: transport should use connection_manager::connect() instead)
	 *
	 * @param cm: associated connection manager
	 * @param handle: handle from listener on a remote node
	 * @param transport_connect_msg:
	 * 	pointer to transport-provided connect message
	 */
	nccl_ofi_cm_send_connector(nccl_ofi_connection_manager *cm,
		nccl_net_ofi_conn_handle handle,
		const void *transport_connect_msg);

	fi_addr_t dest_addr;

	void set_conn_resp_msg(const nccl_ofi_cm_conn_msg &conn_resp_msg);

	void set_conn_msg_delivered() {
		conn_msg_delivered = true;
	}

	/* Back-pointer to connection manager */
	nccl_ofi_connection_manager *cm;

	std::vector<uint8_t> conn_msg;
	std::vector<uint8_t> conn_resp_msg;
	nccl_ofi_freelist_elem_t *send_elem;
	nccl_ofi_cm_send_conn_req send_conn_req;
	std::optional<nccl_ofi_cm_conn_msg> received_conn_resp_msg;

	bool conn_msg_sent;
	bool conn_msg_delivered;

	uint32_t s_comm_id;

	void prepare_conn_msg(nccl_net_ofi_conn_handle *handle, nccl_ofi_cm_conn_msg *conn_msg);

	friend class nccl_ofi_connection_manager;
};

/**
 * Connection manager. Represents state of the connection management code.
 *
 * Intent is for client code to store and initialize a connection manager per
 * (transport-specific) domain.
 *
 * The CM code maintains a separate Libfabric endpoint and state that will be
 * shared across all connections created using this connection manager instance.
 * The created endpoint will be bound to the caller-supplied completion queue.
 */
class nccl_ofi_connection_manager
{
public:
	/**
	 * Initializes the CM system state. Creates an endpoint and posts
	 * initial buffer pool
	 *
	 * @param info, domain:
	 *      Libfabric info and domain objects against which the CM endpoint
	 *      will be created
	 *
	 * @param cq:
	 *      the completion queue to bind the new endpoint to. Ops submitted
	 *      through the CM code will have a context pointer to
	 *      nccl_net_ofi_context_t, with appropriate completion handling
	 *      functions
	 *
	 * @param mr_key_pool:
	 *      caller's mr_key_pool associated with domain. This ensures CM's
	 *      memory registrations unique MR keys that don't conflict with
	 *      other parts of the code
	 *
	 * @param conn_msg_size:
	 *      size of transport-specific part of connect and connect response
	 *      messages
	 */
	nccl_ofi_connection_manager(fi_info *info, fid_domain *domain, fid_cq *cq,
				    nccl_ofi_idpool_t &mr_key_pool, size_t conn_msg_size);

	/**
	 * Destructor. Finalizes CM endpoint and other state.
	 *
	 * Note: when the connection manager is destroyed, all associated
	 * listeners and connectors are invalidated.
	 */
	~nccl_ofi_connection_manager();

	/**
	 * Create a new listener to accept connections
	 *
	 * @param transport_select_conn_resp_callback:
	 *      Caller (transport) - provided callback to provide the
	 *      transport-specific connect response message. The transport may
	 *      select information for the connect response message based on the
	 *      connect message from sender (e.g., endpoint-per-comm mode)
	 *
	 * 	See doc above on transport_select_conn_resp_fn for parameters
	 *
	 * @param transport_select_conn_resp_input:
	 * 	Transport-selected state to be provided to the callback
	 */
	std::unique_ptr<nccl_ofi_cm_listener> listen(
		transport_select_conn_resp_fn transport_select_conn_resp_callback,
		void *transport_select_conn_resp_input);

	/**
	 * Establish a new connection to the listener identified by handle
	 *
	 * Returns a connector object that can be used to query for completion
	 * of the connection and obtain the response from the receiver/
	 *
	 * @param handle: handle from listener on a remote node
	 * @param transport_connect_msg:
	 * 	Connect message. This should point to a buffer of size
	 * 	conn_msg_size (parameter to constructor)
	 */
	std::unique_ptr<nccl_ofi_cm_send_connector> connect(
		nccl_net_ofi_conn_handle handle,
		const void *transport_connect_msg);
private:

	/* TODO store cm endpoint here */

	nccl_ofi_idpool_t listener_id_pool;
	nccl_ofi_idpool_t connector_id_pool;

	size_t conn_msg_size;
};


#endif /* NCCL_OFI_CM_H_ */
