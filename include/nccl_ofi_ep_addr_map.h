/*
 * Copyright (c) 2024 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#ifndef NCCL_OFI_EP_ADDR_MAP_H
#define NCCL_OFI_EP_ADDR_MAP_H

#include "nccl_ofi.h"

/**
 * A linked list of pairs of (ep, HashSet<addr>). The list should be stored in
 * the calling code as a pointer to this struct, initialized to NULL.
 */
struct ep_pair_list_elem;
typedef struct ep_pair_list_elem ep_pair_list_elem_t;

/**
 * Find endpoint in the list ep_pair_list that is not already connected to addr.
 * If all endpoints are already connected to addr, return NULL.
 *
 * @param ep_pair_list list of eps and addresses
 * @param addr Libfabric address of size MAX_EP_ADDR
 */
nccl_net_ofi_ep_t *nccl_ofi_get_ep_for_addr(ep_pair_list_elem_t *ep_pair_list, void *addr);

/**
 * Add ep to the list ep_pair_list, with a single connection to addr.
 *
 * @param ep_pair_list list of eps and addresses
 * @param ep pointer to endpoint
 * @param addr Libfabric address of size MAX_EP_ADDR
 */
void nccl_ofi_insert_ep_for_addr(ep_pair_list_elem_t *ep_pair_list, nccl_net_ofi_ep_t *ep, void *addr);

/**
 * Remove ep from the list ep_pair_list, if present
 *
 * @param ep_pair_list list of eps and addresses
 * @param ep pointer to endpoint
 */
void nccl_ofi_delete_ep_for_addr(ep_pair_list_elem_t *ep_pair_list, nccl_net_ofi_ep_t *ep);

#endif
