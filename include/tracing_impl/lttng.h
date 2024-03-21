/*
 * Copyright (c) 2022 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#pragma once

#if HAVE_LTTNG_UST

#undef LTTNG_UST_TRACEPOINT_PROVIDER
#define LTTNG_UST_TRACEPOINT_PROVIDER nccl_ofi_plugin

#undef LTTNG_UST_TRACEPOINT_INCLUDE
#define LTTNG_UST_TRACEPOINT_INCLUDE "include/tracing_impl/lttng.h"

/*
 * To add a tracepoint at the nccl_ofi_plugin layer:
 * Add a definition of LTTNG_UST_TRACEPOINT_EVENT.
 * LTTNG_UST_TRACEPOINT_EVENT(
 *      nccl_ofi_plugin,
 *      <NewTracepointName>,
 *      LTTNG_UST_TP_ARGS(
 *          <type1>, <arg1>,
 *          <type2>, <arg2>
 *      ),
 *      LTTNG_UST_TP_FIELDS(
 *          lttng_ust_field_integer(<type1>, name1, <arg1>)
 *          lttng_ust_field_integer(<type2>, name2, <arg2>)
 *      )
 * )
 *
 * <NewTracepointName> will appear as the tracepoint name in the
 * tracing output, and arguments <arg1> and <arg2> with <name1> and
 * <name2> will appear in that trace as data.
 *
 * Add a macro to the top level tracing.h
 *
 */

/*
 * LTTNG_UST_TRACEPOINT_HEADER_MULTI_READ must be included so that the tracepoints
 * can be defined and compiled from tracepoint.c, and so they can be referenced
 * from any other files.
 *
 */

#if !defined(NCCL_OFI_TRACEPOINT_H) || defined(LTTNG_UST_TRACEPOINT_HEADER_MULTI_READ)
#define NCCL_OFI_TRACEPOINT_H

#include <lttng/tracepoint.h>

LTTNG_UST_TRACEPOINT_EVENT(
    nccl_ofi_plugin,
    Send,
    LTTNG_UST_TP_ARGS(
            int, dev,
            int, size,
            void *, comm,
            uint16_t, msg_seq_num,
            void *, request,
            void *, nccl_req
    ),
    LTTNG_UST_TP_FIELDS(
            lttng_ust_field_integer(int, dev, dev)
            lttng_ust_field_integer(int, size, size)
            lttng_ust_field_integer_hex(uint64_t, comm, (uint64_t)comm)
            lttng_ust_field_integer(uint16_t, msg_seq_num, msg_seq_num)
            lttng_ust_field_integer_hex(uint64_t, request, (uint64_t)request)
            lttng_ust_field_integer_hex(uint64_t, nccl_req, (uint64_t)nccl_req)
    )
)



LTTNG_UST_TRACEPOINT_EVENT(
    nccl_ofi_plugin,
    Send_ctrl_recv,
    LTTNG_UST_TP_ARGS(
            int, dev,
            int, rail_id,
            void *, comm,
            uint16_t, msg_seq_num
    ),
    LTTNG_UST_TP_FIELDS(
            lttng_ust_field_integer(int, dev, dev)
            lttng_ust_field_integer(int, rail_id, rail_id)
            lttng_ust_field_integer_hex(uint64_t, comm, (uint64_t)comm)
            lttng_ust_field_integer(uint16_t, msg_seq_num, msg_seq_num)
    )
)




LTTNG_UST_TRACEPOINT_EVENT(
    nccl_ofi_plugin,
    Send_write_segment_start,
    LTTNG_UST_TP_ARGS(
            int, dev,
            int, rail_id,
            size_t, size,
            void *, comm,
            uint16_t, msg_seq_num,
            void *, request
    ),
    LTTNG_UST_TP_FIELDS(
            lttng_ust_field_integer(int, dev, dev)
            lttng_ust_field_integer(int, rail_id, rail_id)
            lttng_ust_field_integer(size_t, size, size)
            lttng_ust_field_integer_hex(uint64_t, comm, (uint64_t)comm)
            lttng_ust_field_integer(uint16_t, msg_seq_num, msg_seq_num)
            lttng_ust_field_integer_hex(uint64_t, request, (uint64_t)request)
    )
)



LTTNG_UST_TRACEPOINT_EVENT(
    nccl_ofi_plugin,
    Send_write_segment_complete,
    LTTNG_UST_TP_ARGS(
            int, dev,
            int, rail_id,
            void *, comm,
            uint16_t, msg_seq_num,
            void *, request
    ),
    LTTNG_UST_TP_FIELDS(
            lttng_ust_field_integer(int, dev, dev)
            lttng_ust_field_integer(int, rail_id, rail_id)
            lttng_ust_field_integer_hex(uint64_t, comm, (uint64_t)comm)
            lttng_ust_field_integer(uint16_t, msg_seq_num, msg_seq_num)
            lttng_ust_field_integer_hex(uint64_t, request, (uint64_t)request)
    )
)



LTTNG_UST_TRACEPOINT_EVENT(
    nccl_ofi_plugin,
    Recv,
    LTTNG_UST_TP_ARGS(
            int, dev,
            int, tag,
            int, size,
            void *, request,
            void *, nccl_req
    ),
    LTTNG_UST_TP_FIELDS(
            lttng_ust_field_integer(int, dev, dev)
            lttng_ust_field_integer(int, tag, tag)
            lttng_ust_field_integer(int, size, size)
            lttng_ust_field_integer_hex(uint64_t, request, (uint64_t)request)
            lttng_ust_field_integer_hex(uint64_t, nccl_req, (uint64_t)nccl_req)
    )
)



LTTNG_UST_TRACEPOINT_EVENT(
    nccl_ofi_plugin,
    Recv_ctrl_send_complete,
    LTTNG_UST_TP_ARGS(
            void *, request
    ),
    LTTNG_UST_TP_FIELDS(
            lttng_ust_field_integer_hex(uint64_t, request, (uint64_t)request)
    )
)



LTTNG_UST_TRACEPOINT_EVENT(
    nccl_ofi_plugin,
    Recv_segment_complete,
    LTTNG_UST_TP_ARGS(
            int, dev,
            int, rail_id,
            size_t, size,
            void *, request
    ),
    LTTNG_UST_TP_FIELDS(
            lttng_ust_field_integer(int, dev, dev)
            lttng_ust_field_integer(int, rail_id, rail_id)
            lttng_ust_field_integer(size_t, size, size)
            lttng_ust_field_integer_hex(uint64_t, request, (uint64_t)request)
    )
)


LTTNG_UST_TRACEPOINT_EVENT(
    nccl_ofi_plugin,
    Eager_recv,
    LTTNG_UST_TP_ARGS(
            int, dev,
            int, rail_id,
            void *, comm,
            uint16_t, msg_seq_num
    ),
    LTTNG_UST_TP_FIELDS(
            lttng_ust_field_integer(int, dev, dev)
            lttng_ust_field_integer(int, rail_id, rail_id)
            lttng_ust_field_integer_hex(uint64_t, comm, (uint64_t)comm)
            lttng_ust_field_integer(uint16_t, msg_seq_num, msg_seq_num)
    )
)


LTTNG_UST_TRACEPOINT_EVENT(
    nccl_ofi_plugin,
    ProcessCompletions,
    LTTNG_UST_TP_ARGS(
            void *, request,
            void *, ctx
    ),
    LTTNG_UST_TP_FIELDS(
            lttng_ust_field_integer_hex(uint64_t, request, (uint64_t)request)
            lttng_ust_field_integer(uint64_t, ctx, (uint64_t)ctx)
    )
)



LTTNG_UST_TRACEPOINT_EVENT(
    nccl_ofi_plugin,
    Flush,
    LTTNG_UST_TP_ARGS(
            void *, request,
            void *, nccl_req
    ),
    LTTNG_UST_TP_FIELDS(
            lttng_ust_field_integer_hex(uint64_t, request, (uint64_t)request)
            lttng_ust_field_integer_hex(uint64_t, nccl_req, (uint64_t)nccl_req)
    )
)


LTTNG_UST_TRACEPOINT_EVENT(
    nccl_ofi_plugin,
    Pending_queue_insert,
    LTTNG_UST_TP_ARGS(
            void *, request
    ),
    LTTNG_UST_TP_FIELDS(
            lttng_ust_field_integer_hex(uint64_t, request, (uint64_t)request)
    )
)


LTTNG_UST_TRACEPOINT_EVENT(
    nccl_ofi_plugin,
    Pending_queue_remove,
    LTTNG_UST_TP_ARGS(
            void *, request
    ),
    LTTNG_UST_TP_FIELDS(
            lttng_ust_field_integer_hex(uint64_t, request, (uint64_t)request)
    )
)

#endif /* NCCL_OFI_TRACEPOINT_H */

#include <lttng/tracepoint-event.h>

#else
#define lttng_ust_tracepoint(...)
#endif