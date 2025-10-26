#include "gin/nccl_ofi_gin_ep.h"
#include "gin/nccl_ofi_gin_reqs.h"

#include "nccl_ofi_cuda.h"
#include "nccl_ofi_ofiutils.h"
#include "nccl_ofi_mr.h"
#include "nccl_ofi_param.h"
#include "nccl_ofi_gin_ep.h"

nccl_ofi_gin_ep_t::nccl_ofi_gin_ep_t(nccl_net_ofi_domain_t *domain_arg) :
	domain(domain_arg),
	rx_buff_fl(nullptr, &freelist_deleter)
{
	auto ofi_domains = domain->get_ofi_domains();
	this->num_rails = ofi_domains.size();

	rails.reserve(this->num_rails);
	control_rails.reserve(this->num_rails);

	constexpr size_t num_buffers = 2048; /* TODO param*/
	assert_always(this->num_rails > 0 && num_buffers % this->num_rails == 0);
	const size_t num_buffers_per_rail = num_buffers / this->num_rails;

	nccl_ofi_freelist_t *rx_buff_fl_tmp = nullptr;
	int ret = nccl_ofi_freelist_init_mr
		(sizeof(nccl_net_ofi_gin_signal_metadata_msg_t),
		 num_buffers * 2 /* x2 for data + ctrl */, 0, num_buffers * 2,
		 nullptr, nullptr, freelist_regmr_host_fn, freelist_deregmr_host_fn, domain,
		 1, &rx_buff_fl_tmp);
	if (ret != 0) {
		throw std::runtime_error("Failed to init rx_buff_fl");
	}
	this->rx_buff_fl.reset(rx_buff_fl_tmp);

	// Create rails
	for (uint16_t r = 0; r < this->num_rails; r++) {
		rails.emplace_back(r, this, num_buffers_per_rail);
		control_rails.emplace_back(r, this, num_buffers_per_rail);
	}

	ret = alloc_write_ack_buffer();
	if (ret != 0) {
		throw std::runtime_error("Failed to alloc write ack buffer");
	}
}

nccl_ofi_gin_ep_t::~nccl_ofi_gin_ep_t()
{
	[[maybe_unused]]
	int ret = close_write_ack_buffer();
	assert(ret == 0);
}

/**
 * Set mr attrs. This function closely resembles the same one in RDMA
 */
static int set_mr_req_attr(uint64_t mr_key,
			   nccl_ofi_mr_ckey_ref ckey, uint64_t *flags,
			   int type, struct fi_mr_attr *mr_attr)
{
	int ret = 0;

	/* Basic put-signal access */
	mr_attr->access = FI_WRITE | FI_REMOTE_WRITE;
	nccl_ofi_mr_ckey_fill_mr_attrs(ckey, mr_attr, flags);

	switch (type) {
	case NCCL_PTR_HOST:
		mr_attr->iface = FI_HMEM_SYSTEM;
		break;
#if HAVE_CUDA
	case NCCL_PTR_CUDA:
		mr_attr->iface = FI_HMEM_CUDA;

		/* Get CUDA device ID */
		ret = nccl_net_ofi_get_cuda_device_for_addr(
			(void*)nccl_ofi_mr_ckey_baseaddr(ckey),
			&mr_attr->device.cuda);
		if (OFI_UNLIKELY(ret != 0)) {
			return ret;
		}
		break;
#endif

	default:
		return -EINVAL;
	}

	mr_attr->requested_key = mr_key;

	return ret;
}


int nccl_ofi_gin_ep_t::reg_mr(nccl_ofi_mr_ckey_ref ckey, int type, nccl_ofi_gin_mr_handle_t **mhandle)
{
	int ret = 0;
	struct fi_mr_attr mr_attr = {};
	uint64_t regattr_flags = 0;
	nccl_ofi_idpool_t *key_pool = this->domain->mr_rkey_pool;

	auto ofi_domains = domain->get_ofi_domains();

	*mhandle = NULL;

	auto ret_handle = std::make_unique<nccl_ofi_gin_mr_handle_t>(num_rails);

	if (key_pool->get_size() != 0) {
		auto key = key_pool->allocate_id();
		if (OFI_UNLIKELY(key == FI_KEY_NOTAVAIL)) {
			NCCL_OFI_WARN("MR key allocation failed");
			return -ENOMEM;
		}
		ret_handle->mr_key = static_cast<uint64_t>(key);
	}

	ret = set_mr_req_attr(ret_handle->mr_key, ckey, &regattr_flags, type, &mr_attr);
	if (OFI_UNLIKELY(ret != 0)) {
		NCCL_OFI_WARN("Could not set registration request attributes, dev: %d",
				domain->get_device()->dev_id);
		return ret;
	}

	/* Register memory on each rail */
	for (uint16_t rail_id = 0; rail_id != num_rails; ++rail_id) {
		auto mr_result = nccl_ofi_ofiutils_mr_regattr(ofi_domains[rail_id],
							      &mr_attr,
							      regattr_flags);
		if (OFI_UNLIKELY(mr_result.is_failure())) {
			NCCL_OFI_WARN("Could not register memory on rail %u with flag %lu",
				      rail_id, regattr_flags);
			return mr_result.error_code;
		}
		ret_handle->mr[rail_id] = std::move(mr_result.resource);
	}

	*mhandle = ret_handle.release();
}


int nccl_ofi_gin_ep_t::dereg_mr(nccl_ofi_gin_mr_handle_t *handle_ptr)
{
	if (OFI_UNLIKELY(handle_ptr == NULL)) {
		return;
	}

	auto *mr_rkey_pool = domain->mr_rkey_pool;

	if (mr_rkey_pool->get_size() != 0) {
		mr_rkey_pool->free_id(handle_ptr->mr_key);
	}

	delete handle_ptr;
}

int nccl_ofi_gin_ep_t::alloc_write_ack_buffer()
{
	// Create write-ack buffer (target of write acks)
	int ret = nccl_net_ofi_alloc_mr_buffer(system_page_size, &write_ack_buffer.addr);
	if (ret != 0) {
		NCCL_OFI_WARN("Failed to allocate write ack buffer; RC: %d", ret);
		return ret;
	}

	auto ckey = nccl_ofi_mr_ckey_mk_vec(write_ack_buffer.addr, system_page_size);

	ret = reg_mr(&ckey, NCCL_PTR_HOST, &write_ack_buffer.mr_handle);
	if (ret != 0) {
		NCCL_OFI_WARN("Failed to register write ack buffer; RC: %d", ret);
		close_write_ack_buffer();
		return ret;
	}
}

static inline struct fi_info *get_rx_cq_info(struct fi_info *info)
{
	/* We need to call fi_getinfo again, but this time pass FI_RX_CQ_DATA */
	ofi_info_ptr rx_cq_info = ofi_info_ptr(fi_dupinfo(info));

	rx_cq_info->mode |= FI_RX_CQ_DATA;
	rx_cq_info->domain_attr->cq_data_size = 4;

	struct fi_info *results = nullptr;
	int ret = fi_getinfo(FI_VERSION(1, 18), nullptr, nullptr, 0ULL, rx_cq_info.get(), &results);
	if (ret != 0) {
		throw std::runtime_error("Failed to get rx_cq_info");
	};

	/* There should only be exactly one result */
	assert_always(results != nullptr);
	assert_always(results->next == nullptr);

	/* Make sure we actually got back the info we wanted */
	assert_always(results->mode & FI_RX_CQ_DATA);
	assert_always(results->domain_attr->cq_data_size == 4);

	return results;
}


nccl_ofi_gin_ep_rail_t::nccl_ofi_gin_ep_rail_t(uint16_t rail_id_, nccl_ofi_gin_ep_t *gin_ep,
					       size_t num_rx_buffers)
					       : rail_id(rail_id_)
					         
{
	auto *domain = gin_ep->domain;
	auto &ofi_domain = domain->get_ofi_domains()[rail_id];

	/* Create cq */
	fi_cq_attr cq_attr = {};
	cq_attr.format = FI_CQ_FORMAT_DATA;
	cq_attr.size = ofi_nccl_cq_size();
	auto cq_result = nccl_ofi_ofiutils_cq_create(ofi_domain, nullptr);
	if (OFI_UNLIKELY(cq_result.is_failure())) {
		NCCL_OFI_WARN("Couldn't open CQ. RC: %d, ERROR: %s",
				cq_result.error_code, fi_strerror(-cq_result.error_code));
		throw std::runtime_error("GIN: ofi cq creation failed");
	}
	this->cq = std::move(cq_result.resource);

	/* Create an av */
	auto av_result = nccl_ofi_ofiutils_av_create(ofi_domain);
	if (av_result.is_failure()) {
		throw std::runtime_error("Failed to create av");
	}

	av = std::move(av_result.resource);

	struct fi_info *info = domain->get_device()->get_ofi_infos()[rail_id];
	ofi_info_ptr rx_cq_info(get_rx_cq_info(info));

	/* Create ep */
	auto ep_result = nccl_ofi_ofiutils_ep_create(rx_cq_info.get(), ofi_domain, av, cq);
	if (ep_result.is_failure()) {
		throw std::runtime_error("Failed to create ep");
	}

	ofi_ep = std::move(ep_result.resource);

	/* Now, create the receive pool for all rails */
	if (num_rx_buffers > 0) {
		recv_reqs.reserve(num_rx_buffers);
		for (size_t i = 0; i < num_rx_buffers; i++) {
			recv_reqs.emplace_back(gin_ep, this);
			int ret = recv_reqs[i].post();
			if (ret != 0) {
				throw std::runtime_error("Failed to post recv req");
			}
		}
	}
}
