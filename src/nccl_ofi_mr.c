/*
 * Copyright (c) 2024 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#include "config.h"

#include <stdlib.h>
#include <errno.h>
#include "nccl_ofi.h"
#include "nccl_ofi_mr.h"
#include "nccl_ofi_pthread.h"

int nccl_ofi_mr_cache_init(nccl_ofi_mr_cache_t **cache, size_t num_entries)
{
	int ret = 0;

	nccl_ofi_mr_cache_t *ret_cache = calloc(1, sizeof(*ret_cache));
	if (!ret_cache) {
		ret = -ENOMEM;
		goto out;
	}

	ret_cache->slots = calloc(num_entries, sizeof(ret_cache->slots));
	if (!ret_cache->slots) {
		NCCL_OFI_WARN("Could not allocate memory for cache slots");
		ret = -ENOMEM;
		goto out;
	}

	ret_cache->size = num_entries;
	ret_cache->used = 0;

	*cache = ret_cache;
out:
	return ret;
}

void nccl_ofi_mr_cache_finalize(nccl_ofi_mr_cache_t *cache)
{
	free(cache->slots);

	free(cache);
}

static int nccl_ofi_mr_cache_grow(nccl_ofi_mr_cache_t *cache)
{
	void *ptr;
	int ret = 0;
	cache->size *= 2;
	NCCL_OFI_TRACE(NCCL_NET, "Growing cache to size %d", cache->size);
	ptr =  realloc(cache->slots, cache->size * sizeof(*cache->slots));
	if (!ptr) {
		NCCL_OFI_WARN("Unable to grow cache");
		ret = -ENOMEM;
		goto out;
	}
	cache->slots = ptr;

out:
	return ret;
}

static inline void compute_page_address(uintptr_t addr, size_t size, uintptr_t *page_addr,
				     size_t *pages)
{
	uintptr_t page_size = (uintptr_t) system_page_size;
	*page_addr = addr & -page_size; /* start of page of data */
	*pages = (addr + size - *page_addr + page_size-1)/page_size; /* Number of pages in buffer */
}

void *nccl_ofi_mr_cache_lookup_entry(nccl_ofi_mr_cache_t *cache,
				     uintptr_t addr,
				     size_t size)
{
	uintptr_t page_addr;
	size_t pages;

	compute_page_address(addr, size, &page_addr, &pages);

	for (int slot = 0;;slot++) {
		if (slot == cache->used || page_addr < cache->slots[slot]->addr) {
			/* cache missed */
			return NULL;
		} else if ((page_addr >=  cache->slots[slot]->addr) &&
                           ((page_addr - cache->slots[slot]->addr)/system_page_size + pages) <=  cache->slots[slot]->pages) {
			/* cache hit */
			NCCL_OFI_TRACE(NCCL_NET, "Found MR handle for %p in cache slot %d", addr, slot);
			cache->slots[slot]->refcnt++;
			return cache->slots[slot]->handle;
		}
	}
}

int nccl_ofi_mr_cache_insert_entry(nccl_ofi_mr_cache_t *cache,
				   uintptr_t addr,
				   size_t size,
				   void *handle)
{
	uintptr_t page_addr;
	size_t pages;
	int ret = 0;

	compute_page_address(addr, size, &page_addr, &pages);

	for (int slot = 0;;slot++) {
		if (slot == cache->used || addr < cache->slots[slot]->addr) {
			/* cache missed */

			/* grow the cache if needed */
			if (cache->used == cache->size) {
				nccl_ofi_mr_cache_grow(cache);
			}

			assert(cache->slots);
			memmove(cache->slots+slot+1, cache->slots+slot, (cache->used - slot) * sizeof(nccl_ofi_reg_entry_t*));
			cache->slots[slot] = calloc(1, sizeof(nccl_ofi_reg_entry_t));

			nccl_ofi_reg_entry_t *entry = cache->slots[slot];

			entry->addr = page_addr;
			entry->pages = pages;
			entry->refcnt = 1;
			entry->handle = handle;

			cache->used++;
		} else if ((addr >=  cache->slots[slot]->addr) &&
                           ((addr - cache->slots[slot]->addr)/system_page_size + pages) <=  cache->slots[slot]->pages) {
			/* cache hit */
			NCCL_OFI_WARN("Entry already exists for addr %p size %zu", addr, size);
			ret = -EEXIST;
			goto out;
		}

	}

out:
	return ret;
}

static int nccl_ofi_mr_cache_lookup_handle(nccl_ofi_mr_cache_t *cache, void *handle)
{
	for (int i = 0; i < cache->used; i++) {
		if (handle == cache->slots[i])
			return i;
	}
	return -1;
}

int nccl_ofi_mr_cache_del_entry(nccl_ofi_mr_cache_t *cache, void *handle)
{
	int slot = -1;
	int ret = 0;

	slot = nccl_ofi_mr_cache_lookup_handle(cache, handle);
	if (slot < 0) {
		NCCL_OFI_WARN("Did not find entry to delete");
		ret = -ENOENT;
		goto out;
	}

	/* Keep entry alive for other users */
	if (--cache->slots[slot]->refcnt) {
		goto out;
	}

	/* Free this entry and defrag cache */
	free(cache->slots[slot]);
	memmove(cache->slots+slot, cache->slots+slot+1, (cache->used-slot-1)*sizeof(nccl_ofi_mr_cache_t*));

	/* Signal to caller to deregister handle */
	ret = 1;

out:
	return ret;
}
