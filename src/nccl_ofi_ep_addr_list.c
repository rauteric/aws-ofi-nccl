/*
 * Copyright (c) 2024 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#include "config.h"

#include "nccl_ofi.h"

#include "nccl_ofi_ep_addr_list.h"
#include "nccl_ofi_pthread.h"

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
typedef struct ep_pair_list_elem {
	pair_ep_addr_set_t pair;
	struct ep_pair_list_elem *prev;
	struct ep_pair_list_elem *next;
} ep_pair_list_elem_t;

/**
 * Outer structure storing the ep list and a mutex to protect access
 */
struct nccl_ofi_ep_addr_list {
	ep_pair_list_elem_t *ep_pair_list;
	pthread_mutex_t mutex;
};

void nccl_ofi_init_ep_addr_list(nccl_ofi_ep_addr_list_t **list)
{
	nccl_ofi_ep_addr_list_t *ret_list = calloc(1, sizeof(*ret_list));
	if (!ret_list) {
		NCCL_OFI_WARN("Failed to allocate list");
		goto error;
	}

	ret_list->ep_pair_list = NULL;

	if (nccl_net_ofi_mutex_init(&ret_list->mutex, NULL) != 0) {
		NCCL_OFI_WARN("Failed to init mutex");
		goto error;
	}

	goto exit;

error:
	if (ret_list) free(ret_list);
	ret_list = NULL;

exit:
	*list = ret_list;
}

nccl_net_ofi_ep_t *nccl_ofi_get_ep_for_addr(nccl_ofi_ep_addr_list_t *ep_list, void *addr)
{
	nccl_net_ofi_mutex_lock(&ep_list->mutex);

	ep_pair_list_elem_t *ep_pair = NULL;

	nccl_net_ofi_ep_t *ret_ep = NULL;

	DL_FOREACH(ep_list->ep_pair_list, ep_pair) {
		hashed_addr_t *found_handle;
		HASH_FIND(hh, ep_pair->pair.addr_set, (char *)addr, MAX_EP_ADDR, found_handle);
		if (found_handle) {
			/* This ep already has a connection to the address, skip to next */
			continue;
		} else {
			/* We found an ep that is not connected to addr, so return it */
			hashed_addr_t *new_addr = malloc(sizeof(hashed_addr_t));
			if (!new_addr) {
				NCCL_OFI_WARN("Failed to allocate new address");
				abort();
			}
			memcpy(&new_addr->addr, addr, MAX_EP_ADDR);
			HASH_ADD(hh, ep_pair->pair.addr_set, addr, MAX_EP_ADDR, new_addr);
			ret_ep = ep_pair->pair.ep;
			goto exit;
		}
	}

exit:
	nccl_net_ofi_mutex_unlock(&ep_list->mutex);

	return ret_ep;
}

void nccl_ofi_insert_ep_for_addr(nccl_ofi_ep_addr_list_t *ep_list, nccl_net_ofi_ep_t *ep, void *addr)
{
	nccl_net_ofi_mutex_lock(&ep_list->mutex);

	hashed_addr_t *new_addr = malloc(sizeof(*new_addr));
	if (!new_addr) {
		NCCL_OFI_WARN("Failed to allocate new address");
		abort();
	}
	memcpy(new_addr->addr, addr, MAX_EP_ADDR);

	ep_pair_list_elem_t *new_pair = malloc(sizeof(*new_pair));
	if (!new_pair) {
		NCCL_OFI_WARN("Failed to allocate new ep list element");
		abort();
	}
	new_pair->pair.ep = ep;
	new_pair->pair.addr_set = NULL;
	HASH_ADD(hh, new_pair->pair.addr_set, addr, MAX_EP_ADDR, new_addr);

	DL_APPEND(ep_list->ep_pair_list, new_pair);

	nccl_net_ofi_mutex_unlock(&ep_list->mutex);
}

static void delete_ep_list_entry(ep_pair_list_elem_t *ep_pair_list, ep_pair_list_elem_t *elem)
{
	hashed_addr_t *e, *tmp;
	/* Delete all addr entries in this ep's hashset */
	HASH_ITER(hh, elem->pair.addr_set, e, tmp) {
		HASH_DEL(elem->pair.addr_set, e);
		free(e);
	}
	DL_DELETE(ep_pair_list, elem);
	free(elem);
}

void nccl_ofi_delete_ep_for_addr(nccl_ofi_ep_addr_list_t *ep_list, nccl_net_ofi_ep_t *ep)
{
	nccl_net_ofi_mutex_lock(&ep_list->mutex);

	ep_pair_list_elem_t *ep_pair, *ep_pair_tmp;
	DL_FOREACH_SAFE(ep_list->ep_pair_list, ep_pair, ep_pair_tmp) {
		if (ep_pair->pair.ep == ep) {
			delete_ep_list_entry(ep_list->ep_pair_list, ep_pair);
			goto exit;
		}
	}

exit:
	nccl_net_ofi_mutex_unlock(&ep_list->mutex);
}
