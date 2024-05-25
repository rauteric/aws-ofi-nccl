/*
 * Copyright (c) 2024 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#include "config.h"

#include "nccl_ofi.h"

#include "nccl_ofi_ep_addr_map.h"

#include "contrib/uthash.h"
#include "contrib/utlist.h"

#include <rdma/fi_domain.h>

typedef struct {
	char addr[MAX_EP_ADDR];
	UT_hash_handle hh;
} hashed_addr_t;

typedef struct {
	nccl_net_ofi_ep_t *ep;
	hashed_addr_t *addr_set;
} pair_ep_addr_set_t;

struct ep_pair_list_elem {
	pair_ep_addr_set_t pair;
	struct ep_pair_list_elem *prev;
	struct ep_pair_list_elem *next;
};

/**
 * Find endpoint in the list ep_pair_list that is not already connected to addr.
 * If all endpoints are already connected to addr, return NULL.
 */
nccl_net_ofi_ep_t *nccl_ofi_get_ep_for_addr(ep_pair_list_elem_t *ep_pair_list, void *addr)
{
	//print_addr("get_ep_for_addr", addr);
	ep_pair_list_elem_t *ep_pair;

	DL_FOREACH(ep_pair_list, ep_pair) {
		hashed_addr_t *found_handle;
		HASH_FIND(hh, ep_pair->pair.addr_set, (char *)addr, MAX_EP_ADDR, found_handle);
		if (found_handle) {
			continue;
		} else {
			/* We found an ep that is not connected to addr, so return it */
			hashed_addr_t *new_addr = malloc(sizeof(hashed_addr_t));
			if (!new_addr) abort();
			memcpy(&new_addr->addr, addr, MAX_EP_ADDR);
			HASH_ADD(hh, ep_pair->pair.addr_set, addr, MAX_EP_ADDR, new_addr);
			return ep_pair->pair.ep;
		}
	}

	/* At this point, we haven't found an endpoint that isn't already connected
	   to addr, so return NULL and let caller create a new one */
	return NULL;
}

/**
 * Add ep to the list ep_pair_list, with a single connection to addr.
 */
void nccl_ofi_insert_ep_for_addr(ep_pair_list_elem_t *ep_pair_list, nccl_net_ofi_ep_t *ep, void *addr) {

	//print_addr("nccl_ofi_insert_ep_for_addr", addr);
	hashed_addr_t *new_addr = malloc(sizeof(*new_addr));
	if (!new_addr) abort();
	memcpy(new_addr->addr, addr, MAX_EP_ADDR);

	ep_pair_list_elem_t *new_pair = malloc(sizeof(*new_pair));
	if (!new_pair) abort();
	new_pair->pair.ep = ep;
	new_pair->pair.addr_set = NULL;
	HASH_ADD(hh, new_pair->pair.addr_set, addr, MAX_EP_ADDR, new_addr);

	DL_APPEND(ep_pair_list, new_pair);
}

/**
 * Remove ep from the list ep_pair_list
 */
void nccl_ofi_delete_ep_for_addr(ep_pair_list_elem_t *ep_pair_list, nccl_net_ofi_ep_t *ep)
{
	ep_pair_list_elem_t *ep_pair, *ep_pair_tmp;
	DL_FOREACH_SAFE(ep_pair_list, ep_pair, ep_pair_tmp) {
		if (ep_pair->pair.ep == ep) {
			hashed_addr_t *e, *tmp;
			HASH_ITER(hh, ep_pair->pair.addr_set, e, tmp) {
				HASH_DEL(ep_pair->pair.addr_set, e);
				free(e);
			}
			DL_DELETE(ep_pair_list, ep_pair);
			free(ep_pair);
			return;
		}
	}
	abort();
}
