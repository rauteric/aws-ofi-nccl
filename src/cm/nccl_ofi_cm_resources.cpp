#include "cm/nccl_ofi_cm_resources.h"

using namespace nccl_ofi_cm;

conn_msg_buffer_manager::conn_msg_buffer_manager(fid_domain *domain, size_t buffer_size);

cm_resources::cm_resources(fi_info *info, fid_domain *domain, fid_cq *cq,
			   nccl_ofi_idpool_t &_mr_key_pool, size_t _conn_msg_data_size) :
	buff_mgr(domain, sizeof(nccl_ofi_cm_conn_msg) + conn_msg_data_size),
	rx_reqs(),
	ep(info, domain),
	listener_map(),
	send_connector_map(),

	conn_msg_data_size(_conn_msg_data_size),
	mr_key_pool(_mr_key_pool),
	next_connector_id(0)
{
	assert(false); /* TODO */
	/* Post buffer pool */
}

