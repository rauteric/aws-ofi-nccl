/*
 * Copyright (c) 2025 Amazon.com, Inc. or its affiliates. All rights reserved.
 */
#include "config.h"

#include <stdexcept>

#include "cm/nccl_ofi_cm.h"
#include "nccl_ofi_ofiutils.h"
#include "nccl_ofi_log.h"

static inline int cm_reg_mr(void *cm_ptr, void *data, size_t size, void **mr_handle)
{
	int ret = 0;
	nccl_ofi_cm_mr_handle *ret_handle = nullptr;
	*mr_handle = nullptr;

	auto cm = static_cast<nccl_ofi_connection_manager *>(cm_ptr);

	fid_domain *domain = cm->get_domain();

	struct fi_mr_attr mr_attr = {};
	struct iovec _iovec = {data, size};
	mr_attr.iov_count = 1;
	mr_attr.mr_iov = &_iovec;
	mr_attr.iface = FI_HMEM_SYSTEM;

	uint64_t regattr_flags = 0;

	/* TODO key pool */

	/* Allocate rdma memory registration handle */
	ret_handle = new nccl_ofi_cm_mr_handle{ };

	mr_attr.access = FI_SEND | FI_RECV;

	ret = fi_mr_regattr(domain, &mr_attr,
			    regattr_flags, &ret_handle->mr);
	if (ret != 0) {
		NCCL_OFI_WARN("CM: Unable to register memory. RC: %d, Error: %s",
			      ret, fi_strerror(-ret));
		delete ret_handle;
		return ret;
	}

	/* TODO fi_endpoint_mr mode */

	*mr_handle = ret_handle;
	return 0;
}

static inline int cm_dereg_mr(void *handle)
{
	auto mr_handle = static_cast<nccl_ofi_cm_mr_handle *>(handle);
	int ret = fi_close(&mr_handle->mr->fid);
	if (ret != 0) {
		NCCL_OFI_WARN("Unable to de-register memory. RC: %d, Error: %s",
			      ret, fi_strerror(-ret));
	}
	delete mr_handle;
	return ret;
}


nccl_ofi_connection_manager::nccl_ofi_connection_manager(fi_info *info, fid_domain *_domain, fid_cq *cq,
							 size_t num_comm_ids)
							 : domain(_domain), conn_msg_fl(),
							   rx_req_list(),
							   l_comm_id_pool(num_comm_ids),
							   data_comm_id_pool(num_comm_ids)
{
	int ret = nccl_ofi_ofiutils_init_connection(info, domain, &ep, &av, &cq);
	if (ret != 0) {
		/* We can't return an error. If not caught, this is going to propagate up and
		 * eventually terminate the program, which may or may not be what we want.
		 * TODO revisit */
		throw std::runtime_error("nccl_ofi_connection_manager: failed call to nccl_ofi_ofiutils_init_connection");
	}
	nccl_ofi_freelist_t *conn_msg_fl_ptr = nullptr;
	ret = nccl_ofi_freelist_init_mr(sizeof(nccl_ofi_cm_conn_msg), 4, 4, 0, nullptr, nullptr,
					    cm_reg_mr, cm_dereg_mr, this, 1, &conn_msg_fl_ptr);
	if (ret != 0) {
		throw std::runtime_error("nccl_ofi_connection_manager: failed to create conn_msg fl");
	}

	conn_msg_fl.reset(conn_msg_fl_ptr);

	set_conn_ep_name();

	post_rx_buffers();
}


nccl_ofi_connection_manager::~nccl_ofi_connection_manager()
{
	/* TODO: the last arg (dev_id = 0) is (usually) wrong, but is only used for a print */
	nccl_ofi_ofiutils_ep_release(ep, av, nullptr, /* dev_id */0);
}


void nccl_ofi_connection_manager::set_conn_ep_name()
{
	conn_ep_name.name_len = sizeof(conn_ep_name.name);

	int ret = fi_getname(&ep->fid, conn_ep_name.name, &conn_ep_name.name_len);
	if (ret == -FI_ETOOSMALL) {
		NCCL_OFI_WARN("Endpoint's address length (%zu) is larger than supplied buffer length (%d)",
			      conn_ep_name.name_len, MAX_EP_ADDR);
		throw std::runtime_error("Failed call to fi_getname");
	} else if (ret != 0) {
		NCCL_OFI_WARN("Call to fi_getname() failed with RC: %d, ERROR: %s",
			      ret, fi_strerror(-ret));
		throw std::runtime_error("Failed call to fi_getname");
	}
}


void nccl_ofi_connection_manager::post_rx_buffers()
{
	/* TODO make this a parameter */
	const size_t num_rx_buffer = 1;
	rx_req_list.reserve(num_rx_buffer);

	for (size_t i = 0; i < num_rx_buffer; ++i) {
		rx_req_list.emplace_back(this);
		int ret = rx_req_list[i].post_rx();
		if (ret == -FI_EAGAIN) {
			/* TODO */ abort();
		} else if (ret != 0) {
			throw std::runtime_error("Failed to post rx buffer");
		}
	}
}


nccl_ofi_cm_l_comm *nccl_ofi_connection_manager::listen()
{
	return new nccl_ofi_cm_l_comm(this);
}


nccl_ofi_cm_l_comm *nccl_ofi_connection_manager::get_l_comm(uint32_t l_comm_id)
{
	auto it = l_comm_map.find(l_comm_id);
	if (it != l_comm_map.end()) {
		return it->second;
	} else {
		return nullptr;
	}
}


nccl_ofi_cm_s_comm *nccl_ofi_connection_manager::get_s_comm(uint32_t s_comm_id)
{
	auto it = s_comm_map.find(s_comm_id);
	if (it != s_comm_map.end()) {
		return it->second;
	} else {
		return nullptr;
	}
}

int nccl_ofi_connection_manager::av_insert_address(ep_name address, fi_addr_t *fi_addr)
{
	int ret = fi_av_insert(av, address, 1, fi_addr, 0, NULL);
	if (OFI_UNLIKELY(ret != 1)) {
		NCCL_OFI_WARN("CM: Unable to insert remote address into address vector "
			      "for device. RC: %s", fi_strerror(-ret));
		return -EINVAL;
	}
	return 0;
}


nccl_ofi_cm_s_comm *nccl_ofi_connection_manager::connect(nccl_ofi_cm_handle *handle,
							 const nccl_ofi_cm_ep_rail_info &rail_info)
{
	nccl_ofi_cm_s_comm *s_comm = new nccl_ofi_cm_s_comm(this, handle, rail_info);

	int ret = this->av_insert_address(handle->name, &s_comm->dest_addr);
	if (ret != 0) {
		throw std::runtime_error("Failed call to av_insert_address");
	}

	return s_comm;
}
