/*
 * Copyright (c) 2025 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#ifndef NCCL_OFI_CM_RESOURCES_H_
#define NCCL_OFI_CM_RESOURCES_H_

#include <rdma/fabric.h>
#include <unordered_map>

#include "cm/nccl_ofi_cm_reqs.h"
#include "cm/nccl_ofi_cm.h"

#include "nccl_ofi_freelist.h"

namespace nccl_ofi_cm {

class conn_msg_buffer_manager
{
public:
	/**
	 * Constructor; allocate pool of connection message buffers
	 */
	conn_msg_buffer_manager(fid_domain *domain, size_t buffer_size);

	/**
	 * Destructor; release pool of buffers
	 */
	~conn_msg_buffer_manager();

private:
	nccl_ofi_freelist_t *buff_fl;
};

/**
 * Encapsulates a Libfabric endpoint for use with the CM
 */
class endpoint
{
public:
	endpoint(fi_info *info, fid_domain *domain);

	~endpoint();
private:
	/* Input to OFI resources */
	fid_domain *domain;

	/* Created by CM */
	fid_ep *ep;
	fid_av *av;

	/* Address of locally owned endpoint */
	fi_addr_t ep_addr;
};

template <typename T>
class connector_id_map
{
public:
	void insert_connector(uint64_t id, T* connector);
	bool get_connector(uint64_t id, T** connector);
private:
	std::unordered_map<uint64_t, T*> map;
};

class cm_resources
{
public:
	/* Members */
	conn_msg_buffer_manager buff_mgr;
	std::vector<nccl_ofi_cm_rx_req> rx_reqs;
	endpoint ep;
	connector_id_map<nccl_ofi_cm_listener> listener_map;
	connector_id_map<nccl_ofi_cm_send_connector> send_connector_map;

	/* Methods */

	cm_resources(fi_info *info, fid_domain *domain, fid_cq *cq,
		nccl_ofi_idpool_t &mr_key_pool, size_t conn_msg_data_size);

	uint64_t get_next_connector_id() { return next_connector_id++; }

private:
	size_t conn_msg_data_size;
	nccl_ofi_idpool_t &mr_key_pool;
	uint64_t next_connector_id;
};

}

#endif /* NCCL_OFI_CM_RESOURCES_H_ */
