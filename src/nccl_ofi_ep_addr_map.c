/*
 * Copyright (c) 2024 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#include "config.h"

#include "nccl_ofi.h"

#include "nccl_ofi_ep_addr_map.h"

#include "contrib/uthash.h"
#include "contrib/utlist.h"

/**
 * A Libfabric address, stored in a form hashable by uthash
 */
typedef struct {
	char addr[MAX_EP_ADDR];
	UT_hash_handle hh;
} hashed_addr_t;

/**
 * A struct storing a pair of (ep, addr_set)
 */
typedef struct {
	nccl_net_ofi_ep_t *ep;
	/* Hash set of addresses the endpoint is connected to */
	hashed_addr_t *addr_set;
} pair_ep_addr_set_t;

/**
 * A linked list of pairs of (ep, HashSet<addr>). The list should be stored in
 * the calling code as a pointer to this struct.
 */
struct ep_pair_list_elem {
	pair_ep_addr_set_t pair;
	struct ep_pair_list_elem *prev;
	struct ep_pair_list_elem *next;
};

nccl_net_ofi_ep_t *nccl_ofi_get_ep_for_addr(ep_pair_list_elem_t *ep_pair_list, void *addr)
{
	ep_pair_list_elem_t *ep_pair;

	DL_FOREACH(ep_pair_list, ep_pair) {
		hashed_addr_t *found_handle;
		HASH_FIND(hh, ep_pair->pair.addr_set, (char *)addr, MAX_EP_ADDR, found_handle);
		if (found_handle) {
			/* This ep already has a connection to the address, skip to next*/
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

void nccl_ofi_insert_ep_for_addr(ep_pair_list_elem_t *ep_pair_list, nccl_net_ofi_ep_t *ep, void *addr) {

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

void nccl_ofi_delete_ep_for_addr(ep_pair_list_elem_t *ep_pair_list, nccl_net_ofi_ep_t *ep)
{
	ep_pair_list_elem_t *ep_pair, *ep_pair_tmp;
	DL_FOREACH_SAFE(ep_pair_list, ep_pair, ep_pair_tmp) {
		if (ep_pair->pair.ep == ep) {
			hashed_addr_t *e, *tmp;
			/* Delete all addr entries in this ep's hashset */
			HASH_ITER(hh, ep_pair->pair.addr_set, e, tmp) {
				HASH_DEL(ep_pair->pair.addr_set, e);
				free(e);
			}
			DL_DELETE(ep_pair_list, ep_pair);
			free(ep_pair);
			return;
		}
	}
}
