/*
 * Copyright (c) 2025 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#ifndef NCCL_OFI_CM_RESOURCES_H_
#define NCCL_OFI_CM_RESOURCES_H_

#include <rdma/fabric.h>

#include <deque>
#include <unordered_map>

#include "cm/nccl_ofi_cm_reqs.h"

#include "nccl_ofi_freelist.h"

namespace nccl_ofi_cm {

/**
 * Encapsulates a Libfabric endpoint for use with the CM
 *
 * Also encapsulates other OFI resources -- domain (from transport)
 * and av
 */
class endpoint
{
public:
	struct mr_handle_t {
		fid_mr *mr;
		uint64_t mr_key;
		endpoint &ep;
	};

	endpoint(fi_info *info, fid_domain *domain, nccl_ofi_idpool_t &mr_key_pool, fid_cq *cq);

	~endpoint();

	int get_ep_address(void *address, size_t &addr_len);

	fi_addr_t av_insert_address(const void *address);

	int post_send(nccl_ofi_freelist_elem_t &send_elem, size_t size,
		      fi_addr_t dest_addr, nccl_ofi_cm_req &req);

	int post_recv(nccl_ofi_freelist_elem_t &recv_elem, size_t size,
		      nccl_ofi_cm_req &req);

	int post_inject(void *send_buffer, size_t size, fi_addr_t dest_addr);

	/**
	 * Close associated ofi_ep, while leaving other resources open
	 */
	int close_ofi_ep();

	/* Menory registration/deregistration. Note: these functions are static
	   to be usable with the freelist interface */
	static int reg_mr(void *ep_ptr, void *data, size_t size, void **mr_handle);

	static int dereg_mr(void *handle_ptr);
private:
	/* Input to CM */
	fid_domain *domain;
	nccl_ofi_idpool_t &mr_key_pool;

	/* Created by CM */
	fid_ep *ep;
	fid_av *av;

	size_t max_inject_size;
};

class conn_msg_buffer_manager
{
public:
	/**
	 * Constructor; allocate and register pool of connection message buffers
	 */
	conn_msg_buffer_manager(endpoint &ep, size_t buffer_size);

	/**
	 * Destructor; release pool of buffers
	 */
	~conn_msg_buffer_manager();

	nccl_ofi_freelist_elem_t &allocate_conn_msg();
	void free_conn_msg(nccl_ofi_freelist_elem_t &conn_msg);

private:
	endpoint &ep;
	nccl_ofi_freelist_t *buff_fl;
};

template <typename T>
class connector_id_map
{
public:
	void insert_connector(uint64_t id, T& connector);
	T& get_connector(uint64_t id);
	void remove_connector(uint64_t id);
private:
	std::unordered_map<uint64_t, T&> map;
};

class pending_requests_queue
{
public:
	void add_req(nccl_ofi_cm_req &req);

	int process_pending_reqs();
private:
	std::deque<nccl_ofi_cm_req *> pending_reqs;
};

class cm_resources
{
public:
	/* Public members */
	endpoint ep;
private:
	size_t conn_msg_data_size;

public:
	conn_msg_buffer_manager buff_mgr;
	connector_id_map<nccl_ofi_cm_listener> listener_map;
	connector_id_map<nccl_ofi_cm_send_connector> send_connector_map;
	pending_requests_queue pending_reqs_queue;

	/* Methods */

	cm_resources(fi_info *info, fid_domain *domain, fid_cq *cq,
		nccl_ofi_idpool_t &mr_key_pool, size_t conn_msg_data_size);

	~cm_resources();

	uint64_t get_next_connector_id() { return next_connector_id++; }

	size_t get_conn_msg_data_size() { return conn_msg_data_size; }
	size_t get_conn_msg_size() { return sizeof(nccl_ofi_cm_conn_msg) + conn_msg_data_size; }

	uint64_t next_connector_id;
	std::vector<nccl_ofi_cm_rx_req *> rx_reqs;
};


/**
 * Define templated member functions
 */

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

	return result->second;
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

}

#endif /* NCCL_OFI_CM_RESOURCES_H_ */
