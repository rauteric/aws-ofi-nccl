/*
 * Copyright (c) 2024 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#ifndef NCCL_OFI_EP_ADDR_MAP_H
#define NCCL_OFI_EP_ADDR_MAP_H

#include "nccl_ofi.h"

struct ep_pair_list_elem;
typedef struct ep_pair_list_elem ep_pair_list_elem_t;

nccl_net_ofi_ep_t *nccl_ofi_get_ep_for_addr(struct ep_pair_list_elem *ep_pair_list, void *addr);

void nccl_ofi_insert_ep_for_addr(struct ep_pair_list_elem *ep_pair_list, nccl_net_ofi_ep_t *ep, void *addr);

void nccl_ofi_delete_ep_for_addr(struct ep_pair_list_elem *ep_pair_list, nccl_net_ofi_ep_t *ep);

#endif
