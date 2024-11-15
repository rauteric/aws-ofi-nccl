/*
 * Copyright (c) 2022-2024 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#ifndef NVTX_H
#define NVTX_H

#if HAVE_NVTX_TRACING
#include <nvtx3/nvToolsExt.h>

#define NCCL_OFI_N_NVTX_DOMAIN_PER_COMM 8

static inline void nvtx_mark_domain(nvtxDomainHandle_t domain, const char* name, uint32_t color)
{
	nvtxEventAttributes_t eventAttrib = {};

	eventAttrib.version = NVTX_VERSION;
	eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
	eventAttrib.colorType = NVTX_COLOR_ARGB;
	eventAttrib.color = color;
	eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
	eventAttrib.message.ascii = name;

	nvtxDomainMarkEx(domain, &eventAttrib);
}

static inline nvtxRangeId_t nvtx_start_domain(bool have_domain, nvtxDomainHandle_t domain, const char* name, uint32_t color) {
	nvtxEventAttributes_t eventAttrib = {};

	eventAttrib.version = NVTX_VERSION;
	eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
	eventAttrib.colorType = NVTX_COLOR_ARGB;
	eventAttrib.color = color;
	eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
	eventAttrib.message.ascii = name;

	if (have_domain)
		return nvtxDomainRangeStartEx(domain, &eventAttrib);
	else
		return nvtxRangeStartEx(&eventAttrib);
}

static inline nvtxRangeId_t nvtx_start(const char* name, uint32_t color) {
	return nvtx_start_domain(false, 0, name, color);
}

static inline void nvtx_end_domain(nvtxDomainHandle_t domain, nvtxRangeId_t id) {
	nvtxDomainRangeEnd(domain, id);
}

static inline void nvtx_end(nvtxRangeId_t id) {
	nvtxRangeEnd(id);
}

#define NCCL_OFI_TRACE_SEND_NVTX(...)
#define NCCL_OFI_TRACE_SEND_END_NVTX(...)
#define NCCL_OFI_TRACE_EAGER_SEND_START_NVTX(...)
#define NCCL_OFI_TRACE_EAGER_SEND_COMPLETE_NVTX(...)
#define NCCL_OFI_TRACE_SEND_CTRL_RECV_NVTX(...)
#define NCCL_OFI_TRACE_SEND_CTRL_START_NVTX(...)
#define NCCL_OFI_TRACE_SEND_CTRL_END_NVTX(...)
#define NCCL_OFI_TRACE_SEND_WRITE_SEG_START_NVTX(...)
#define NCCL_OFI_TRACE_SEND_WRITE_SEG_COMPLETE_NVTX(...)
#define NCCL_OFI_TRACE_RECV_NVTX(...)
#define NCCL_OFI_TRACE_RECV_END_NVTX(...)
#define NCCL_OFI_TRACE_RECV_SEGMENT_COMPLETE_NVTX(...)
#define NCCL_OFI_TRACE_EAGER_RECV_NVTX(...)
#define NCCL_OFI_TRACE_FLUSH_NVTX(...)
#define NCCL_OFI_TRACE_READ_NVTX(...)
#define NCCL_OFI_TRACE_WRITE_NVTX(...)
#define NCCL_OFI_TRACE_PENDING_INSERT_NVTX(...)
#define NCCL_OFI_TRACE_PENDING_REMOVE_NVTX(...)

#else

#define NCCL_OFI_TRACE_SEND_NVTX(...)
#define NCCL_OFI_TRACE_SEND_END_NVTX(...)
#define NCCL_OFI_TRACE_EAGER_SEND_START_NVTX(...)
#define NCCL_OFI_TRACE_EAGER_SEND_COMPLETE_NVTX(...)
#define NCCL_OFI_TRACE_SEND_CTRL_RECV_NVTX(...)
#define NCCL_OFI_TRACE_SEND_CTRL_START_NVTX(...)
#define NCCL_OFI_TRACE_SEND_CTRL_END_NVTX(...)
#define NCCL_OFI_TRACE_SEND_WRITE_SEG_START_NVTX(...)
#define NCCL_OFI_TRACE_SEND_WRITE_SEG_COMPLETE_NVTX(...)
#define NCCL_OFI_TRACE_RECV_NVTX(...)
#define NCCL_OFI_TRACE_RECV_END_NVTX(...)
#define NCCL_OFI_TRACE_RECV_SEGMENT_COMPLETE_NVTX(...)
#define NCCL_OFI_TRACE_EAGER_RECV_NVTX(...)
#define NCCL_OFI_TRACE_FLUSH_NVTX(...)
#define NCCL_OFI_TRACE_READ_NVTX(...)
#define NCCL_OFI_TRACE_WRITE_NVTX(...)
#define NCCL_OFI_TRACE_PENDING_INSERT_NVTX(...)
#define NCCL_OFI_TRACE_PENDING_REMOVE_NVTX(...)

#endif

#endif /* NVTX_H */
