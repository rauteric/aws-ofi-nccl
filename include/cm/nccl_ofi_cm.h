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

class freelist_deleter
{
public:
	void operator()(nccl_ofi_freelist_t *freelist)
	{
		nccl_ofi_freelist_fini(freelist);
	}
};

class nccl_ofi_connection_manager
{
public:
	nccl_ofi_connection_manager(fi_info *info, fid_domain *_domain,
				    fid_cq *cq, size_t num_comm_ids,
				    nccl_ofi_idpool_t *_mr_key_pool);
	~nccl_ofi_connection_manager();

	nccl_ofi_cm_l_comm *listen();

	nccl_ofi_cm_s_comm *connect(nccl_ofi_cm_handle *handle,
				    const nccl_ofi_cm_ep_rail_info &rail_info);

	nccl_ofi_cm_l_comm *get_l_comm(uint32_t l_comm_id);
	nccl_ofi_cm_s_comm *get_s_comm(uint32_t s_comm_id);

	nccl_ofi_freelist_elem_t *alloc_conn_msg()
	{
		return nccl_ofi_freelist_entry_alloc(conn_msg_fl.get());
	}
	void free_conn_msg(nccl_ofi_freelist_elem_t *conn_msg)
	{
		nccl_ofi_freelist_entry_free(conn_msg_fl.get(), conn_msg);
	}

	fid_ep *get_ep() {return ep;}
	fid_domain *get_domain() {return domain;}

	int av_insert_address(ep_name address, fi_addr_t *fi_addr);

	const cm_ep_name &get_conn_ep_name() {return conn_ep_name;}

	nccl_ofi_idpool_t *get_l_comm_id_pool() { return &l_comm_id_pool; }
	nccl_ofi_idpool_t *get_data_comm_id_pool() { return &data_comm_id_pool; }

	std::unordered_map<uint32_t, nccl_ofi_cm_l_comm *> *get_l_comm_map()
	{ return &l_comm_map; }

	std::unordered_map<uint32_t, nccl_ofi_cm_s_comm *> *get_s_comm_map()
	{ return &s_comm_map; }

	nccl_ofi_idpool_t *get_mr_key_pool() {return mr_key_pool;}

private:
	/* Input */
	fid_domain *domain;
	/* Created by CM */
	fid_ep *ep;
	fid_av *av;

	std::unordered_map<uint32_t, nccl_ofi_cm_l_comm *> l_comm_map;
	std::unordered_map<uint32_t, nccl_ofi_cm_s_comm *> s_comm_map;

	std::unique_ptr<nccl_ofi_freelist_t, freelist_deleter> conn_msg_fl;

	/* This must appear after conn_msg_fl so it is destructed first... */
	std::vector<nccl_ofi_cm_rx_req> rx_req_list;

	nccl_ofi_idpool_t l_comm_id_pool;
	nccl_ofi_idpool_t data_comm_id_pool;

	nccl_ofi_idpool_t *mr_key_pool;

	cm_ep_name conn_ep_name;

	void set_conn_ep_name();

	void post_rx_buffers();
};

#endif /* NCCL_OFI_CM_H_ */
