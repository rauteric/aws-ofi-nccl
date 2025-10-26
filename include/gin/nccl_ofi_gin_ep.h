/*
 * Copyright (c) 2025      Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#ifndef NCCL_OFI_GIN_EP_H
#define NCCL_OFI_GIN_EP_H

#include "rdma/fabric.h"
#include <vector>

#include "nccl_ofi_freelist.h"
#include "gin/nccl_ofi_gin_types.h"
#include "gin/nccl_ofi_gin_reqs.h"

static inline void freelist_deleter(nccl_ofi_freelist_t *fl)
{
	int ret = nccl_ofi_freelist_fini(fl);
	if (ret != 0) {
		NCCL_OFI_WARN("Failed to finalize freelist");
		assert(false); abort();
	}
}

class nccl_ofi_gin_mr_handle_t : public nccl_net_ofi_mr_handle_t
{
public:
	nccl_ofi_gin_mr_handle_t(size_t num_rails) : nccl_net_ofi_mr_handle_t(0),
						     mr(num_rails)
	{
	}

	/**
	 * @brief	Get first MR key for GIN MR handle
	 * 		This interface isn't necessary for GIN.
	 */
	int get_mr_key(uint64_t *mr_key_ptr) override { return -ENOTSUP; }

	/* Array of size `num_rails' */
	std::vector<ofi_mr_ptr> mr;
};

struct nccl_ofi_gin_ep_rail_t {

	nccl_ofi_gin_ep_rail_t(uint16_t rail_id_, nccl_ofi_gin_ep_t *gin_ep,
			       size_t num_rx_buffers);

	/* No explicit destructor needed -- resources should clean themselves up */

	const uint16_t rail_id;

	ofi_cq_ptr cq;

	/* Address vector handle */
	ofi_av_ptr av;

	/* Local libfabric endpoint handle */
	ofi_ep_ptr ofi_ep;

	/* RX buffers for control rails */
	std::vector<nccl_net_ofi_gin_recv_req_t> recv_reqs;
};


struct nccl_ofi_gin_ep_t {
	nccl_ofi_gin_ep_t(nccl_net_ofi_domain_t *domain_arg);

	~nccl_ofi_gin_ep_t();

	nccl_net_ofi_domain_t *domain;

	size_t num_rails;

	std::unique_ptr<nccl_ofi_freelist_t, decltype(&freelist_deleter)> rx_buff_fl;

	std::vector<nccl_ofi_gin_ep_rail_t> rails;
	std::vector<nccl_ofi_gin_ep_rail_t> control_rails;

	int reg_mr(nccl_ofi_mr_ckey_ref ckey, int type, nccl_ofi_gin_mr_handle_t **mhandle);

	int dereg_mr(nccl_ofi_gin_mr_handle_t *handle_ptr);

	nccl_ofi_gin_comm& get_comm(uint32_t comm_id) {
		assert(comm_id < gin_comms.size());
		return gin_comms[comm_id];
	}

	void set_comm(uint32_t comm_id, const nccl_ofi_gin_comm& comm) {
		assert(comm_id < gin_comms.size());
		gin_comms[comm_id] = comm;
	}

	void *get_write_ack_buffer_addr() { return write_ack_buffer.addr; }
	nccl_ofi_gin_mr_handle_t *get_write_ack_buffer_mr_handle() { return write_ack_buffer.mr_handle; }

private:
	struct {
		void *addr = nullptr;
		nccl_ofi_gin_mr_handle_t *mr_handle = nullptr;
	} write_ack_buffer;

	std::vector<nccl_ofi_gin_comm&> gin_comms;

	int alloc_write_ack_buffer();
	int close_write_ack_buffer();
};

#endif
