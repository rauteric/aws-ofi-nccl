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
 * An object returned from listener::accept() which represents a connection in
 * progress.
 */
class nccl_ofi_cm_receiver
{
public:
	/**
	 * Return transport-specific connect message data from sender, after
	 * connection is established
	 *
	 * Note: Returns a pointer to memory owned by this object. The memory is
	 * valid until this object is destroyed.
	 */
	const void *get_conn_msg_data()
	{
		return user_conn_msg_data.data();
	}

	/**
	 * Set transport-specific data to be sent in the connect response message
	 *
	 * @param data: transport-provided buffer of size conn_msg_size
	 */
	void set_conn_resp_msg_data(const void *data);

	/**
	 * Test whether the connection is complete. This will return ready=true
	 * when the connect response message has been delivered and the
	 * connection is ready to use
	 *
	 * @param ready: set to true when connection is complete
	 * @return: negative errno code on network-related error
	 */
	int test_ready(bool *ready);

	~nccl_ofi_cm_receiver();

private:
	/**
	 * Construct a receiver. Transport should use listener::accept() instead
	 * of this constructor.
	 */
	nccl_ofi_cm_receiver(nccl_ofi_cm_listener &listener, void *conn_msg);
	std::vector<uint8_t> user_conn_msg_data;
	std::vector<uint8_t> user_conn_resp_msg;

	friend class nccl_ofi_cm_listener;
};

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
	 * The caller (transport) should return this handle back to NCCL (see
	 * the handle argument to NCCL's listen/connect APIs) to be delivered
	 * out-of-band to the remote (send side) node
	 */
	nccl_net_ofi_conn_handle get_handle() { return handle; }

	/**
	 * Accept an incoming connection from the send side to this listener
	 * This returns a nccl_ofi_cm_receiver object that can be used to send
	 * the connect response message to the sender.
	 *
	 * If no connection is ready, returns nullptr.
	 *
	 * See documentation above on nccl_ofi_cm_receiver
	 * 
	 * Note: the caller takes ownership of the memory associated with this
	 * object and should release it by deleting the pointer.
	 */
	nccl_ofi_cm_receiver *accept();

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
	 */
	nccl_ofi_cm_listener(nccl_ofi_connection_manager &cm);

	nccl_ofi_connection_manager &cm;
	uint32_t listener_id;
	nccl_net_ofi_conn_handle handle;
	std::deque<nccl_ofi_cm_receiver&> ready_receiver_queue;

	friend class nccl_ofi_connection_manager;
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
	 * @return: negative errno code on network-related error
	 */
	int test_ready(bool *ready);

	/**
	 * Pointer to transport-specific data returned from receiver side in the
	 * connect response message, once connection is complete. This returns a
	 * pointer to memory owned by this object, and is invalidated when
	 * send_connector is destroyed.
	 */
	const void *get_conn_resp_msg();

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

	void set_conn_resp_msg_data(const void *conn_resp_msg_data);

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

	void prepare_conn_msg(nccl_net_ofi_conn_handle &handle, nccl_ofi_cm_conn_msg &conn_msg);

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
	 *      memory registrations use unique MR keys that don't conflict with
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
	 */
	nccl_ofi_cm_listener* listen();

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
	nccl_ofi_cm_send_connector* connect(nccl_net_ofi_conn_handle handle,
		const void *transport_connect_msg);
private:

	/* TODO store cm endpoint here */

	/**
	 * ID pools to manage CM-internal listener and connector IDs. These are
	 * transparent to the transport and are not related to any transport-
	 * chosen IDs
	 */
	nccl_ofi_idpool_t listener_id_pool;
	nccl_ofi_idpool_t connector_id_pool;

	size_t conn_msg_size;
};


#endif /* NCCL_OFI_CM_H_ */
