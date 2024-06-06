/*
 * Copyright (c) 2024 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#ifndef NCCL_OFI_EP_ADDR_LIST_H
#define NCCL_OFI_EP_ADDR_LIST_H

#include "nccl_ofi.h"

struct ep_addr_list;
typedef struct ep_addr_list ep_addr_list_t;

/**
 * Initialize an endpoint-address-set pair list
 *
 * NOTE: there is no destroy function because this list is stored with device,
 * which has no destructor.
 *
 * @param list: output, pointer to the new list
 */
void nccl_ofi_init_ep_addr_list(ep_addr_list_t **list);

/**
 * Find endpoint in the list ep_pair_list that is not already connected to addr.
 * If all endpoints are already connected to addr, return NULL.
 *
 * @param ep_list list of eps and addresses
 * @param addr Libfabric address of size MAX_EP_ADDR
 */
nccl_net_ofi_ep_t *nccl_ofi_get_ep_for_addr(ep_addr_list_t *ep_list, void *addr);

/**
 * Add ep to the list ep_pair_list, with a single connection to addr.
 *
 * @param ep_list list of eps and addresses
 * @param ep pointer to endpoint
 * @param addr Libfabric address of size MAX_EP_ADDR
 */
void nccl_ofi_insert_ep_for_addr(ep_addr_list_t *ep_list, nccl_net_ofi_ep_t *ep, void *addr);

/**
 * Remove ep from the list ep_pair_list, if present
 *
 * @param ep_list list of eps and addresses
 * @param ep pointer to endpoint
 */
void nccl_ofi_delete_ep_for_addr(ep_addr_list_t *ep_list, nccl_net_ofi_ep_t *ep);

#endif
