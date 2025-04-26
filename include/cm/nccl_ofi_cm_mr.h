/*
 * Copyright (c) 2025 Amazon.com, Inc. or its affiliates. All rights reserved.
 */
#ifndef NCCL_OFI_CM_MR_H_
#define NCCL_OFI_CM_MR_H_

#include <rdma/fabric.h>

#include "cm/nccl_ofi_cm.h"
#include "nccl_ofi_idpool.h"

#define MR_KEY_INIT_VALUE FI_KEY_NOTAVAIL

namespace nccl_ofi_cm {

struct mr_args {
	fid_domain *domain;
	nccl_ofi_idpool_t &mr_key_pool;
};

struct mr_handle {
	fid_mr *mr;
	uint64_t mr_key;
	mr_args args;
};

static inline int cm_dereg_mr(void *handle_ptr)
{
	int ret = 0;
	auto handle = static_cast<mr_handle *>(handle);

	if (handle->args.mr_key_pool.get_size() != 0 &&
			OFI_LIKELY(handle->mr_key != MR_KEY_INIT_VALUE)) {

		handle->args.mr_key_pool.free_id(mr_handle->mr_key);
	}

	if (handle.mr) {
		ret = fi_close(&handle.mr->fid);
		if (ret != 0) {
			NCCL_OFI_WARN("Unable to de-register memory. RC: %d, Error: %s",
				      ret, fi_strerror(-ret));
		}
	}

	delete handle;
	return ret;
}


static inline int cm_reg_mr(void *args_ptr, void *data, size_t size, void **mr_handle)
{
	int ret = 0;
	nccl_ofi_cm_mr_handle *ret_handle = nullptr;
	*mr_handle = nullptr;

	auto args = static_cast<mr_args *>(args_ptr);

	fid_domain *domain = args->domain;

	struct fi_mr_attr mr_attr = {};
	struct iovec _iovec = {data, size};
	mr_attr.iov_count = 1;
	mr_attr.mr_iov = &_iovec;
	mr_attr.iface = FI_HMEM_SYSTEM;

	uint64_t regattr_flags = 0;

	/* Allocate cm memory registration handle */
	ret_handle = new mr_handle { };
	ret_handle->cm = cm;
	ret_handle->mr_key = MR_KEY_INIT_VALUE;

	mr_attr.access = FI_SEND | FI_RECV;

	if (cm->get_mr_key_pool()->get_size() != 0) {
		size_t key = cm->get_mr_key_pool()->allocate_id();
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
		ret = fi_mr_bind(ret_handle->mr, &cm->get_ep()->fid, 0);
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
		cm_dereg_mr(ret_handle);
		ret_handle = nullptr;
	}
	*mr_handle = nullptr;
	return ret;
}

}

#endif /* NCCL_OFI_CM_MR_H_ */
