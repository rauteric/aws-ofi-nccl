/*
 * Copyright (c) 2023 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#include "config.h"

#include <stdio.h>

#include "test-common.h"
#include "nccl_ofi_ep_addr_list.h"

int main(int argc, char *argv[])
{
	ofi_log_function = logger;

	const size_t num_addr = 10;

	nccl_ofi_ep_addr_list_t *ep_addr_list = NULL;

	nccl_ofi_init_ep_addr_list(&ep_addr_list);
	if (!ep_addr_list) {
		NCCL_OFI_WARN("Init ep addr list failed");
		exit(1);
	}

	/** Test insertion and retrieval **/
	for (int i = 0; i < num_addr; ++i) {
		char addr[MAX_EP_ADDR];
		memset(&addr, 0, MAX_EP_ADDR);
		*((uintptr_t *)&addr) = i;
		nccl_net_ofi_ep_t *ep = nccl_ofi_get_ep_for_addr(ep_addr_list, &addr);
		if (i == 0) {
			if (ep) {
				NCCL_OFI_WARN("Ep unexpectedly returned");
				exit(1);
			}
			nccl_ofi_insert_ep_for_addr(ep_addr_list, (nccl_net_ofi_ep_t *)(1), &addr);
		} else {
			if (!ep) {
				NCCL_OFI_WARN("No ep returned when expected");
				exit(1);
			}
			if ((uintptr_t)ep != 1) {
				NCCL_OFI_WARN("Unexpected ep returned");
			}
		}
	}

	/** And again! **/
	for (int i = 0; i < num_addr; ++i) {
		char addr[MAX_EP_ADDR];
		memset(&addr, 0, MAX_EP_ADDR);
		*((uintptr_t *)&addr) = i;
		nccl_net_ofi_ep_t *ep = nccl_ofi_get_ep_for_addr(ep_addr_list, &addr);
		if (i == 0) {
			if (ep) {
				NCCL_OFI_WARN("Ep unexpectedly returned");
				exit(1);
			}
			nccl_ofi_insert_ep_for_addr(ep_addr_list, (nccl_net_ofi_ep_t *)(2), &addr);
		} else {
			if (!ep) {
				NCCL_OFI_WARN("No ep returned when expected");
				exit(1);
			}
			if ((uintptr_t)ep != 2) {
				NCCL_OFI_WARN("Unexpected ep returned");
			}
		}
	}

	/** Test delete **/
	int r = 0;
	r = nccl_ofi_delete_ep_for_addr(ep_addr_list, (nccl_net_ofi_ep_t *)1);
	if (r) {
		NCCL_OFI_WARN("Delete ep failed unexpectedly");
		exit(1);
	}
	r = nccl_ofi_delete_ep_for_addr(ep_addr_list, (nccl_net_ofi_ep_t *)2);
	if (r) {
		NCCL_OFI_WARN("Delete ep failed unexpectedly");
		exit(1);
	}
	r = nccl_ofi_delete_ep_for_addr(ep_addr_list, (nccl_net_ofi_ep_t *)3); // (Doesn't exist)
	if (r != -ENOENT) {
		NCCL_OFI_WARN("Delete ep succeeded unexpectedly");
		exit(1);
	}

	/** Now try retrieving for a new address -- should return NULL since we removed all eps **/
	{
		char addr[MAX_EP_ADDR];
		memset(&addr, 0, MAX_EP_ADDR);
		*((uintptr_t *)&addr) = num_addr+1;
		nccl_net_ofi_ep_t *ep = nccl_ofi_get_ep_for_addr(ep_addr_list, &addr);
		if (ep) {
			NCCL_OFI_WARN("Ep unexpectedly returned");
			exit(1);
		}
	}

	nccl_ofi_ep_addr_list_fini(ep_addr_list);

	printf("Test completed successfully!\n");

	return 0;
}
