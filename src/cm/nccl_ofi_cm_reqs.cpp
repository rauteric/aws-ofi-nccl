/*
 * Copyright (c) 2025 Amazon.com, Inc. or its affiliates. All rights reserved.
 */
#include "config.h"

#include <stdexcept>

#include "nccl_ofi.h"

#include "cm/nccl_ofi_cm_reqs.h"
#include "cm/nccl_ofi_cm.h"

using namespace nccl_ofi_cm;


static inline int cm_req_handle_cq_entry(nccl_net_ofi_context_t *ctx,
					 struct fi_cq_entry *cq_entry_base,
					 uint16_t rail_id)
{
	nccl_ofi_cm_req *req = cpp_container_of(ctx, &nccl_ofi_cm_req::ctx);

	return req->handle_completion();
}

static inline int cm_req_handle_error_entry(nccl_net_ofi_context_t *ctx,
					    struct fid_cq *cq,
					    struct fi_cq_err_entry *err_entry,
					    uint16_t rail_id)
{
	int ret = 0;

	if (err_entry->err == FI_ECANCELED) {
		/* Closing an EP with posted receives will (erroneously) generate
		   cancellation events for the posted receives with the EFA provider
		   in Libfabric versions prior to 1.22. These events are harmless
		   and can be ignored.

		   With Libfabric 1.22 and later, we shouldn't get these cancel
		   events at all. The plugin does not explicitly call fi_cancel. */
		return 0;
	}

	assert(ctx);
	nccl_ofi_cm_req *req = cpp_container_of(ctx, &nccl_ofi_cm_req::ctx);

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


nccl_ofi_cm_req::nccl_ofi_cm_req()
{
	ctx.handle_cq_entry = cm_req_handle_cq_entry;
	ctx.handle_error_entry = cm_req_handle_error_entry;
}


nccl_ofi_cm_rx_req::~nccl_ofi_cm_rx_req()
{
	resources.buff_mgr.free_conn_msg(rx_elem);
}

int nccl_ofi_cm_rx_req::progress()
{
	return resources.ep.post_recv(rx_elem, resources.get_conn_msg_size(), *this);
}


int nccl_ofi_cm_send_conn_req::progress()
{
	return resources.ep.post_send(send_elem, resources.get_conn_msg_size(), dest_addr, *this);
}

int nccl_ofi_cm_send_conn_req::handle_completion()
{
	NCCL_OFI_INFO(NCCL_INIT, "Send completion");
	done_callback();
	/* Free this request resources */
	delete this;
	return 0;
}

nccl_ofi_cm_send_conn_resp_req::nccl_ofi_cm_send_conn_resp_req
	(cm_resources &_resources, fi_addr_t _dest_addr,
	 std::function<void()> _done_callback) :

	resources(_resources),
	send_elem(resources.buff_mgr.allocate_conn_msg()),
	dest_addr(_dest_addr),
	done_callback(_done_callback)
{
	/**
	 * We use a different behavior depending on the value of
	 * data_progress_auto
	 *
	 * When data_progress_auto is true (i.e., we are using a provider
	 * supporting FI_PROGRESS_AUTO), we modify connection establishment
	 * behavior to support NCCL's shared-comm/multi-recv behavior.
	 *
	 * Briefly, in shared-comm/multi-recv mode, NCCL will try to establish
	 * multiple communicators in parallel, and use the first one(s) that
	 * succeed. For plugin's conenction establishment, the resulting
	 * requirement is summarized as: after sending the connect response
	 * message (which is the final message of connection establishment), the
	 * plugin must return a valid recv comm, not NULL.
	 *
	 * To support this, we use fi_inject for the connection response message
	 * so that we don't have to process the resulting completion. We rely on
	 * the provider making progress on the request without NCCL calling into
	 * the plugin to poll the completion queue.
	 *
	 * When we don't have auto progress support, we maintain the old
	 * behavior that returns NULL from accept() until the connect response
	 * message is delivered. This approach will lead to mismatched
	 * communicators and deadlock in shared-comm/multi-recv mode.
	 */
	use_inject = data_progress_auto;
}

nccl_ofi_cm_send_conn_resp_req::~nccl_ofi_cm_send_conn_resp_req()
{
	resources.buff_mgr.free_conn_msg(send_elem);
}

int nccl_ofi_cm_send_conn_resp_req::progress()
{
	if (use_inject) {
		int ret = resources.ep.post_inject(send_elem.ptr, resources.get_conn_msg_size(),
						   dest_addr);
		if (ret == 0) {
			/* Immediately complete request and destroy resources */
			return this->handle_completion();
		}

		return ret;
	} else {
		return resources.ep.post_send(send_elem, resources.get_conn_msg_size(),
					      dest_addr, *this);
	}
}


int nccl_ofi_cm_send_conn_resp_req::handle_completion()
{
	NCCL_OFI_INFO(NCCL_INIT, "Send completion");
	done_callback();
	/* Free this request resources */
	delete this;
	return 0;
}


int nccl_ofi_cm_rx_req::handle_completion()
{
	NCCL_OFI_INFO(NCCL_INIT, "Recv completion");
	nccl_ofi_cm_conn_msg *conn_msg = static_cast<nccl_ofi_cm_conn_msg *>(rx_elem.ptr);
	switch(conn_msg->type) {
	case nccl_ofi_cm_conn_msg::SEND_CONN_MSG: {

		nccl_ofi_cm_listener &listener =
			resources.listener_map.get_connector(conn_msg->remote_id);

		listener.process_conn_msg(*conn_msg);
		break;
	}
	case nccl_ofi_cm_conn_msg::SEND_CONN_RESP_MSG: {

		nccl_ofi_cm_send_connector &connector =
			resources.send_connector_map.get_connector(conn_msg->remote_id);

		connector.process_conn_resp_msg(*conn_msg);
		break;
	}

	}

	/* Repost buffer */
	return this->progress();
}
