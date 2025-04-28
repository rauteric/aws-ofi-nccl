#include "cm/nccl_ofi_cm_resources.h"

#include "nccl_ofi_ofiutils.h"

using namespace nccl_ofi_cm;


endpoint::endpoint(fi_info *info, fid_domain *_domain, nccl_ofi_idpool_t &_mr_key_pool,
		   fid_cq *cq) :
	domain(_domain),
	mr_key_pool(_mr_key_pool),
	max_inject_size(info->tx_attr->inject_size)
{
	NCCL_OFI_INFO(NCCL_INIT, "max inject size: %zu", max_inject_size);
	int ret = nccl_ofi_ofiutils_init_connection(info, domain, &this->ep, &this->av, &cq);
	if (ret != 0) {
		/* We can't return an error. If not caught, this is going to propagate up and
		 * eventually terminate the program, which may or may not be what we want.
		 * TODO revisit */
		throw std::runtime_error("endpoint: failed call to nccl_ofi_ofiutils_init_connection");
	}
}


endpoint::~endpoint()
{
	/* TODO: the last arg (dev_id = 0) is (usually) wrong, but is only used for a print */
	nccl_ofi_ofiutils_ep_release(ep, av, nullptr, /* dev_id */0);
}


int endpoint::get_ep_address(void *address, size_t &addr_len)
{
	int ret = fi_getname(&ep->fid, address, &addr_len);
	if (ret == -FI_ETOOSMALL) {
		NCCL_OFI_WARN("Endpoint's address length (%zu) is larger than supplied buffer length",
			      addr_len);
	} else if (ret != 0) {
		NCCL_OFI_WARN("Call to fi_getname() failed with RC: %d, ERROR: %s",
			      ret, fi_strerror(-ret));
	}

	return ret;
}


fi_addr_t endpoint::av_insert_address(const void *address)
{
	fi_addr_t ret_addr;
	int ret = fi_av_insert(av, address, 1, &ret_addr, 0, NULL);
	if (OFI_UNLIKELY(ret != 1)) {
		NCCL_OFI_WARN("CM: Unable to insert remote address into address vector "
			      "for device.");
		throw std::runtime_error("Failed call to fi_av_insert");
	}
	return ret_addr;
}


conn_msg_buffer_manager::conn_msg_buffer_manager(endpoint &_ep, size_t buffer_size) :
	ep(_ep)
{
	int ret = nccl_ofi_freelist_init_mr(buffer_size, 16, 16, 0, nullptr, nullptr, endpoint::reg_mr, endpoint::dereg_mr,
					    this, 1, &buff_fl);
	if (ret != 0) {
		throw std::runtime_error("Failed to init freelist");
	}
}


conn_msg_buffer_manager::~conn_msg_buffer_manager()
{
	int ret = nccl_ofi_freelist_fini(buff_fl);
	/* Shouldn't throw from destructors, so an assert will do. */
	assert(ret == 0);
}


nccl_ofi_freelist_elem_t &conn_msg_buffer_manager::allocate_conn_msg()
{
	return *(nccl_ofi_freelist_entry_alloc(buff_fl));
}


void conn_msg_buffer_manager::free_conn_msg(nccl_ofi_freelist_elem_t &conn_msg)
{
	nccl_ofi_freelist_entry_free(buff_fl, &conn_msg);
}


template <typename T>
void connector_id_map<T>::insert_connector(uint64_t id, T& connector)
{
	auto result = map.emplace(id, connector);
	if (result.second == false) {
		NCCL_OFI_WARN("Attempt to insert duplicate id");
		throw std::runtime_error("duplicate id insert");
	}
}


template <typename T>
T& connector_id_map<T>::get_connector(uint64_t id)
{
	auto result = map.find(id);

	if (result == map.end()) {
		NCCL_OFI_WARN("Lookup of invalid id");
		throw std::runtime_error("invalid id lookup");
	}
}


template <typename T>
void connector_id_map<T>::remove_connector(uint64_t id)
{
	size_t n_removed = map.erase(id);
	if (n_removed != 1) {
		NCCL_OFI_WARN("Failed to remove connector id: %lu", id);
		throw std::runtime_error("id removal fail");
	}
}


void pending_requests_queue::add_req(nccl_ofi_cm_req &req)
{
	pending_reqs.push_back(req);
}


int pending_requests_queue::process_pending_reqs()
{
	for (auto it = pending_reqs.begin(); it != pending_reqs.end(); ) {
		nccl_ofi_cm_req &req = *it;

		int ret = req.progress();
		if (ret == -FI_EAGAIN) {
			/* Leave req in the queue for next try */
			break;
		} else if (ret == 0) {
			it = pending_reqs.erase(it);
		} else {
			return ret;
		}
	}

	return 0;
}


cm_resources::cm_resources(fi_info *info, fid_domain *domain, fid_cq *cq,
			   nccl_ofi_idpool_t &mr_key_pool, size_t _conn_msg_data_size) :
	ep(info, domain, mr_key_pool, cq),
	buff_mgr(ep, sizeof(nccl_ofi_cm_conn_msg) + conn_msg_data_size),
	listener_map(),
	send_connector_map(),
	pending_reqs_queue(),
	rx_reqs(),

	conn_msg_data_size(_conn_msg_data_size),
	next_connector_id(0)
{
	/* TODO make param */
	const size_t num_rx_reqs = 1;

	rx_reqs.reserve(num_rx_reqs);
	for (size_t i = 0; i < num_rx_reqs; ++i) {
		rx_reqs.push_back(new nccl_ofi_cm_rx_req(*this));
		int ret = rx_reqs[i]->progress();
		if (ret == -FI_EAGAIN) {
			pending_reqs_queue.add_req(*(rx_reqs[i]));
		} else if (ret != 0) {
			throw std::runtime_error("Failed to post rx buffer");
		}
	}
}

cm_resources::~cm_resources()
{
	/* Resources can be destructed in the usual reverse-order, with one exception:
	   The endpoint must be closed first, since posted buffers and requests cannot
	   be freed until the endpoint is closed.
	 */
	int ret = ep.close_ofi_ep();
	if (ret != 0) {
		NCCL_OFI_WARN("Failed to close OFI endpoint: %d", ret);
	}

	/* Free all requests. (A unique_ptr would be better here so these can be freed
	   automatically) */
	for (auto &req : rx_reqs) {
		delete req;
		req = nullptr;
	}
	rx_reqs.clear();
}

#define MR_KEY_INIT_VALUE FI_KEY_NOTAVAIL

int endpoint::dereg_mr(void *handle_ptr)
{
	int ret = 0;
	auto handle = static_cast<mr_handle_t *>(handle_ptr);

	if (handle->ep.mr_key_pool.get_size() != 0 &&
			OFI_LIKELY(handle->mr_key != MR_KEY_INIT_VALUE)) {

		handle->ep.mr_key_pool.free_id(handle->mr_key);
	}

	if (handle->mr) {
		ret = fi_close(&handle->mr->fid);
		if (ret != 0) {
			NCCL_OFI_WARN("Unable to de-register memory. RC: %d, Error: %s",
				      ret, fi_strerror(-ret));
		}
	}

	delete handle;
	return ret;
}

int endpoint::reg_mr(void *ep_ptr, void *data, size_t size, void **mr_handle)
{
	int ret = 0;
	*mr_handle = nullptr;

	auto ep = static_cast<endpoint *>(ep_ptr);

	fid_domain *domain = ep->domain;

	struct fi_mr_attr mr_attr = {};
	struct iovec _iovec = {data, size};
	mr_attr.iov_count = 1;
	mr_attr.mr_iov = &_iovec;
	mr_attr.iface = FI_HMEM_SYSTEM;

	uint64_t regattr_flags = 0;

	/* Allocate cm memory registration handle */
	struct mr_handle_t *ret_handle = new mr_handle_t { nullptr, MR_KEY_INIT_VALUE, *ep};

	mr_attr.access = FI_SEND | FI_RECV;

	if (ep->mr_key_pool.get_size() != 0) {
		size_t key = ep->mr_key_pool.allocate_id();
		if (OFI_UNLIKELY(key == FI_KEY_NOTAVAIL)) {
			NCCL_OFI_WARN("MR key allocation failed");
			ret = -ENOMEM;
			goto error;
		}
		ret_handle->mr_key = static_cast<uint64_t>(key);
		mr_attr.requested_key = ret_handle->mr_key;
	}

	ret = fi_mr_regattr(domain, &mr_attr,
			    regattr_flags, &ret_handle->mr);
	if (ret != 0) {
		NCCL_OFI_WARN("CM: Unable to register memory. RC: %d, Error: %s",
			      ret, fi_strerror(-ret));
		goto error;
	}

	if (endpoint_mr) {
		ret = fi_mr_bind(ret_handle->mr, &ep->ep->fid, 0);
		if (OFI_UNLIKELY(ret != 0)) {
			NCCL_OFI_WARN("CM: Unable to bind MR to EP. RC: %d, Error: %s",
				      ret, fi_strerror(-ret));
			goto error;
		}

		ret = fi_mr_enable(ret_handle->mr);
		if (OFI_UNLIKELY(ret != 0)) {
			NCCL_OFI_WARN("CM: Unable to enable MR. RC: %d, Error: %s",
				       ret, fi_strerror(-ret));
			goto error;
		}
	}

	*mr_handle = ret_handle;
	return 0;
error:
	if (ret_handle) {
		dereg_mr(ret_handle);
		ret_handle = nullptr;
	}
	*mr_handle = nullptr;
	return ret;
}

int endpoint::post_send(nccl_ofi_freelist_elem_t &send_elem, size_t size,
			fi_addr_t dest_addr, nccl_ofi_cm_req &req)
{
	auto mr_handle = static_cast<mr_handle_t *>(send_elem.mr_handle);
	void *desc = fi_mr_desc(mr_handle->mr);

	ssize_t ret = fi_send(ep, send_elem.ptr, size, desc,
			      dest_addr, &req.ctx.ofi_ctx);
	if (ret != 0 && ret != -FI_EAGAIN) {
		NCCL_OFI_WARN("Error in call to fi_send. RC: %zd, Error: %s",
				ret, fi_strerror(-ret));
		return static_cast<int>(ret);
	} else {
		NCCL_OFI_INFO(NCCL_INIT, "Post send");
	}

	return static_cast<int>(ret);
}

int endpoint::post_recv(nccl_ofi_freelist_elem_t &recv_elem, size_t size,
			nccl_ofi_cm_req &req)
{
	auto mr_handle = static_cast<mr_handle_t *>(recv_elem.mr_handle);
	void *desc = fi_mr_desc(mr_handle->mr);

	ssize_t ret = fi_recv(ep, recv_elem.ptr, size, desc,
			      FI_ADDR_UNSPEC, &req.ctx.ofi_ctx);
	if (ret != 0 && ret != -FI_EAGAIN) {
		NCCL_OFI_WARN("Error posting rx buffer. RC: %zd, Error: %s",
			      ret, fi_strerror(-ret));
		return static_cast<int>(ret);
	}

	return static_cast<int>(ret);
}


int endpoint::post_inject(void *send_buffer, size_t size, fi_addr_t dest_addr)
{
	if (size > max_inject_size) {
		NCCL_OFI_WARN("Attempt to inject buffer larger than max_inject_size (%zu)",
			      max_inject_size);
		return -EINVAL;
	}
	ssize_t ret = fi_inject(ep, send_buffer, size, dest_addr);
	if (ret != 0 && ret != -FI_EAGAIN) {
		NCCL_OFI_WARN("Error injecting message. RC: %zd, Error: %s",
			      ret, fi_strerror(-ret));
	}

	return static_cast<int>(ret);
}


int endpoint::close_ofi_ep()
{
	if (ep == nullptr) {
		NCCL_OFI_WARN("ep was already closed");
		return -EINVAL;
	}

	int ret = fi_close(&ep->fid);
	ep = nullptr;
	return ret;
}
