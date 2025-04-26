/*
 * Copyright (c) 2025 Amazon.com, Inc. or its affiliates. All rights reserved.
 */
#include "config.h"

#include <stdexcept>

#include "nccl_ofi.h"
#include "nccl_ofi_ofiutils.h"
#include "nccl_ofi_log.h"

#include "cm/nccl_ofi_cm.h"
#include "cm/nccl_ofi_cm_mr.h"

nccl_ofi_connection_manager::nccl_ofi_connection_manager
	(fi_info *info, fid_domain *domain, fid_cq *cq,
	 nccl_ofi_idpool_t &mr_key_pool, size_t conn_msg_data_size) :

	resources(info, domain, cq, mr_key_pool, conn_msg_data_size)
{
}


nccl_ofi_cm_send_connector* nccl_ofi_connection_manager::connect
	(nccl_net_ofi_conn_handle handle,
	 const void *transport_connect_msg)
{
	return nccl_ofi_cm_send_connector(resources, handle, transport_connect_msg);
}

nccl_ofi_cm_listener::nccl_ofi_cm_listener(nccl_ofi_cm::cm_resources &_resources) :
	resources(_resources)
{ }


nccl_ofi_cm_listener* nccl_ofi_connection_manager::listen()
{
	return new nccl_ofi_cm_listener(resources);
}
