/*
 * Copyright (c) 2025      Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#ifndef NCCL_OFI_GIN_H
#define NCCL_OFI_GIN_H

#include <deque>

#include <gdrapi.h>

#include "nccl_ofi.h"
#include "nccl_ofi_rdma.h"
#include "nccl_ofi_freelist.h"

#include "gin/nccl_ofi_gin_types.h"
#include "gin/nccl_ofi_gin_resources.h"

struct nccl_ofi_gin_listen_comm
{
	int dev;
	nccl_net_ofi_domain_t *domain;
	nccl_net_ofi_ep_t *ep;
	nccl_net_ofi_listen_comm_t *l_comm;
};

/**
 * Represents data associated with a single peer rank
 */
struct nccl_ofi_gin_rank_comm
{
	/* Remote comm id */
	uint32_t comm_id;
	/* Rail addresses */
	fi_addr_t control_address[MAX_NUM_RAILS];
	fi_addr_t address[MAX_NUM_RAILS];

	/* Signal acks are zero-byte RDMA writes (with imm data)
	   sent from receiver to sender. These writes are required to
	   target a valid buffer on both sender and receiver, so this
	   address tracks the empty buffer for each rank */
	uint64_t write_ack_buff_addr;
	uint64_t write_ack_buff_mr_key[MAX_NUM_RAILS];

	/* A sequence number, stored at initiator, exclusively for this (target) peer rank.
	   This allows the remote rank to enforce ordering of signal delivery

	   A 16-bit integer is large enough to store all outstanding requests, because the
	   plugin and NCCL limit max inflight requests. (See NCCL_OFI_MAX_REQUESTS.) */
	uint16_t next_target_seq_num;

	/**
	 * Next-to-be-delivered sequence number, stored at target, from
	 * (initiator) peer rank
	 */
	uint16_t next_delivered_signal_seq_num;

	/* Flag, stored at initiator, indicating the given sequence number (mod
	   max_requests) is in use at initiator side. This allows initiator to
	   track in-use sequence numbers to avoid overflow and only mark
	   iputSignal complete when it has received the ack from the target,
	   which has delivered the signal atomic.
	   */
	bool active_put_signal[NCCL_OFI_MAX_REQUESTS];
};


struct gin_remote_mr
{
	uintptr_t address;
	int num_rails;
	uint64_t mr_key[MAX_NUM_RAILS];
};

struct gin_sym_mr_handle
{
	/* Local address of memory registration */
	void *addr;
	size_t size;

	/* Handle to local memory registration */
	nccl_ofi_gin_mr_handle_t *local_handle;
	/* Type of registration (NCCL_PTR_HOST, NCCL_PTR_CUDA) */
	int type;
	/* GDRCopy handle */
	gdr_mh_t gdr_mr_handle;
	/* Host-mapped pointer to this memory from GDRCopy */
	void *host_map;
	/* Original GDRCopy page-aligned mapped pointer (used for gdr_unmap) */
	void *gdr_mapped_ptr;
	size_t gdr_reglen;

	/* Remote MR information for each peer rank */
	std::vector<gin_remote_mr> remote_mr;
};


/**
 * This represents the main GIN communicator
 */
class nccl_ofi_gin_comm
{
public:
	nccl_ofi_gin_resources resources;

	uint32_t local_comm_id;

	int rank;
	int nranks;

	/* AG ring */
	nccl_net_ofi_send_comm_t *s_comm;
	nccl_net_ofi_recv_comm_t *r_comm;

	/* Remote comm info book */
	std::vector<nccl_ofi_gin_rank_comm> rank_comms;

	/* For each rail (ctrl and data), map of fi_addr => peer comm rank */
	std::unordered_map<fi_addr_t, uint64_t> ctrl_rank_map[MAX_NUM_RAILS];
	std::unordered_map<fi_addr_t, uint64_t> rank_map[MAX_NUM_RAILS];

	/* Map of <rank, msg_seq_num> => recv_req
	 *
	 * This key is guaranteed to be unique because each initiating rank
	 * maintains a monotonically increasing sequence counter for each target
	 * rank. */
	std::unordered_map<uint64_t, nccl_net_ofi_gin_iputsignal_recv_req *>
		outstanding_iput_signal_recv_reqs;

	/* Number of outstanding RDMA writes for signal delivery acknowledgement
	   Used to wait for remaining acknowledgements on communicator close. */
	size_t outstanding_ack_counter;

	std::unique_ptr<nccl_ofi_freelist_t, decltype(&freelist_deleter)> metadata_fl;

	/* Map from pointers to memory registration handle. Used to look up
	   GDRCopy handle for signal delivery.

	   TODO: we could also just pass this in the handle to avoid a map
	   lookup. Not sure yet if that is the right thing to do. */
	std::unordered_map<void *, gin_sym_mr_handle *> mr_handle_map;

	/* For rail scheduling. Currently we do round-robin among rails.
	   TODO:
	   This should be a scheduler object stored with the domain. (The
	   current domain scheduler will do striping for large messages, but
	   here we want a round-robin-only scheduler.) */
	uint16_t next_rail_id;

	/* Pointer to the context's GDRCopy handle (created during initialization) */
	gdr_t gdr_handle;

	nccl_ofi_gin_comm(nccl_net_ofi_domain_t &domain_arg, int dev_id_,
			  int rank_,
			  int nranks_,
			  nccl_net_ofi_send_comm_t *s_comm_,
			  nccl_net_ofi_recv_comm_t *r_comm_,
			  gdr_t gdr_handle_);

	~nccl_ofi_gin_comm()
	{
		if (!this->called_close) {
			int ret = this->close();
			if (ret != 0) {
				NCCL_OFI_WARN("Failed to close GIN comm %d", ret);
			}
		}
	}

	std::deque<nccl_net_ofi_gin_op_req_t *> pending_requests;

	/* Progress the completion queue and try posting any pending requests */
	int progress();

	/* Close the GIN comm, waiting for any outstanding requests as necessary. */
	int close();

private:
	int process_pending_reqs();

	bool called_close = false;
};

int gin_connect(nccl_ofi_gin_ctx* gin_ctx, nccl_net_ofi_conn_handle_t* handles[],
		int nranks, int rank, nccl_ofi_gin_listen_comm* gin_l_comm,
		nccl_ofi_gin_comm** gin_comm_out);

int nccl_ofi_gin_allgather(struct nccl_ofi_gin_comm *comm, void *data, size_t size);

int gin_regMrSymDmaBuf(nccl_ofi_gin_comm* comm, nccl_ofi_mr_ckey_ref ckey, void *data_ptr,
		       size_t size, int type, uint64_t mrFlags, gin_sym_mr_handle** mr_handle_out);

int gin_deregMrSym(nccl_ofi_gin_comm* comm, gin_sym_mr_handle* mr_handle);

int gin_iputSignal(nccl_ofi_gin_comm* gin_comm, uint64_t srcOff, gin_sym_mr_handle* srcMhandle,
		   size_t size, uint64_t dstOff, gin_sym_mr_handle* dstMhandle,
		   uint32_t rank, uint64_t signalOff, gin_sym_mr_handle* signalMhandle,
		   uint64_t signalValue, uint32_t signalOp, nccl_net_ofi_req_t** request);

int gin_handle_signal_metadata_completion(nccl_ofi_gin_comm *gin_comm, fi_addr_t src_addr, uint16_t rail_id,
					  const nccl_net_ofi_gin_signal_metadata_msg_t *metadata_msg);

int gin_handle_signal_write_completion(nccl_ofi_gin_comm *gin_comm, fi_addr_t src_addr, uint16_t rail_id,
				       uint16_t msg_seq_num, uint64_t total_segms, size_t len);

#endif
