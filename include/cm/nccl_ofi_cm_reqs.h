/*
 * Copyright (c) 2025 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#ifndef NCCL_OFI_CM_REQS_H_
#define NCCL_OFI_CM_REQS_H_

#include "cm/nccl_ofi_cm_types.h"
#include "cm/nccl_ofi_cm_resources.h"
#include "nccl_ofi_freelist.h"

namespace nccl_ofi_cm {

/**
 * Note: requests are not to be used directly by the transport; only by the
 * CM code
 */

/**
 * Base class for requests
 */
class nccl_ofi_cm_req
{
public:
	nccl_ofi_cm_req();
	nccl_net_ofi_context_t ctx;

	virtual int handle_completion() = 0;
	virtual int progress() = 0;

	/**
	 * This abstract base class cannot be constructed, but one could try to
	 * destruct it using a pointer to this base class. Since the destructor
	 * is not virtual, prevent this by making the destructor protected.
	 */
protected:
	~nccl_ofi_cm_req() = default;
};

/**
 * Requests for rx buffers
 */
class nccl_ofi_cm_rx_req : public nccl_ofi_cm_req
{
public:
	/**
	 * Constructor. Frees the freelist elem back to the given cm.
	 */
	nccl_ofi_cm_rx_req(cm_resources &_resources) :
		resources(_resources),
		rx_elem(resources.buff_mgr.allocate_conn_msg())
		{ }

	/**
	 * Destructor. Frees the freelist elem.
	 */
	~nccl_ofi_cm_rx_req();

	virtual int handle_completion();
	virtual int progress();

private:

	cm_resources &resources;
	nccl_ofi_freelist_elem_t &rx_elem;
};

/**
 * Send connect message request. Member of send_connector.
 */
class nccl_ofi_cm_send_conn_req : public nccl_ofi_cm_req
{
public:

	nccl_ofi_cm_send_conn_req(cm_resources &_resources, fi_addr_t _dest_addr,
				  std::function<void()> _done_callback) :
		resources(_resources),
		send_elem(resources.buff_mgr.allocate_conn_msg()),
		dest_addr(_dest_addr),
		done_callback(_done_callback)
	{ }

	/**
	 * Destructor. Frees the freelist elem.
	 */
	~nccl_ofi_cm_send_conn_req();

	nccl_ofi_cm_conn_msg &get_conn_msg()
	{
		return *static_cast<nccl_ofi_cm_conn_msg*>(send_elem.ptr);
	};

	virtual int handle_completion();
	virtual int progress();
private:
	cm_resources &resources;
	nccl_ofi_freelist_elem_t &send_elem;
	fi_addr_t dest_addr;
	std::function<void()> done_callback;
};

/**
 * Send connect response message request. Member of receiver.
 */
class nccl_ofi_cm_send_conn_resp_req : public nccl_ofi_cm_req
{
public:

	nccl_ofi_cm_send_conn_resp_req(cm_resources &_resources, fi_addr_t _dest_addr,
				       std::function<void()> _done_callback);

	/**
	 * Destructor. Frees the freelist elem.
	 */
	~nccl_ofi_cm_send_conn_resp_req();

	nccl_ofi_cm_conn_msg &get_conn_resp_msg()
	{
		return *static_cast<nccl_ofi_cm_conn_msg*>(send_elem.ptr);
	};

	virtual int handle_completion();
	virtual int progress();

private:
	bool use_inject;
	cm_resources &resources;
	nccl_ofi_freelist_elem_t &send_elem;
	fi_addr_t dest_addr;
	std::function<void()> done_callback;
};

}

#endif /* NCCL_OFI_CM_REQS_H_ */
