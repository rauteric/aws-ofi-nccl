/*
 * Copyright (c) 2025      Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#ifndef NCCL_OFI_GIN_EP_H
#define NCCL_OFI_GIN_EP_H

#include "rdma/fabric.h"
#include <vector>
#include <unordered_map>

#include "nccl_ofi_freelist.h"
#include "gin/nccl_ofi_gin_types.h"
#include "gin/nccl_ofi_gin_reqs.h"

static inline void freelist_deleter(nccl_ofi_freelist_t *fl)
{
	int ret = nccl_ofi_freelist_fini(fl);
	if (ret != 0) {
		NCCL_OFI_WARN("Failed to finalize freelist");
		assert(false);
	}
}

class nccl_ofi_gin_mr_handle_t : public nccl_net_ofi_mr_handle_t
{
private:
	/* Back-pointer to GIN ep */
	nccl_ofi_gin_ep_t *ep;
public:
	nccl_ofi_gin_mr_handle_t(size_t num_rails, uint64_t mr_key_arg) :
		nccl_net_ofi_mr_handle_t(mr_key_arg), mr(num_rails)
	{
	}

	~nccl_ofi_gin_mr_handle_t()
	{
		auto *mr_rkey_pool = ep->domain->mr_rkey_pool;

		if (mr_rkey_pool->get_size() != 0) {
			mr_rkey_pool->free_id(this->mr_key);
		}
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
			       ofi_cq_ptr &cq, size_t num_rx_buffers);

	/* No explicit destructor needed -- resources should clean themselves up */

	const uint16_t rail_id;

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

	void dereg_mr(nccl_ofi_gin_mr_handle_t *handle_ptr);

	nccl_ofi_gin_comm& get_comm(uint32_t comm_id) {
		auto it = gin_comms.find(comm_id);
		if (it == gin_comms.end()) {
			NCCL_OFI_WARN("Invalid comm_id %d", comm_id);
			throw std::runtime_error("Failed to find comm_id");
		}

		return *(it->second);
	}

	void set_comm(uint32_t comm_id, nccl_ofi_gin_comm& comm) {
		auto it = gin_comms.insert({comm_id, &comm});
		if (!it.second) {
			NCCL_OFI_WARN("Failed to insert duplicate comm_id %d", comm_id);
			throw std::runtime_error("Failed to insert comm_id");
		}
	}

	void *get_write_ack_buffer_addr() { return write_ack_buffer.addr; }
	nccl_ofi_gin_mr_handle_t *get_write_ack_buffer_mr_handle() { return write_ack_buffer.mr_handle; }

	int process_cq();

private:
	int gin_process_completions(struct fi_cq_data_entry *cq_entry,
				    fi_addr_t *src_addrs,
				    uint64_t num_cqes,
				    uint16_t rail_id);

	int gin_process_error_entry(struct fi_cq_err_entry *err_entry,
				    struct fid_cq *cq,
				    uint16_t rail_id);

	int gin_process_cq_rail(uint16_t rail_id);

private:
	struct {
		void *addr = nullptr;
		nccl_ofi_gin_mr_handle_t *mr_handle = nullptr;
	} write_ack_buffer;

	std::unordered_map<uint32_t, nccl_ofi_gin_comm*> gin_comms;

	static ofi_cq_ptr create_cq(ofi_domain_ptr &ofi_domain);

	std::vector<ofi_cq_ptr> rail_cq;

	int alloc_write_ack_buffer();
	int close_write_ack_buffer();
};

#endif
