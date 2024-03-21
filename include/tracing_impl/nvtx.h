/*
 * Copyright (c) 2022-2024 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#pragma once
#if HAVE_NVTX_TRACING
#include "nvToolsExt.h"
static inline void nvtx_push(const char *name) { (void)name; }
static inline void nvtx_pop(void) { }

static inline nvtxRangeId_t nvtx_start(const char* name, uint32_t color) {
	const nvtxEventAttributes_t eventAttrib = {
		.version = NVTX_VERSION,
		.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE,
		.colorType = NVTX_COLOR_ARGB,
		.color = color,
		.messageType = NVTX_MESSAGE_TYPE_ASCII,
		.message = { .ascii = name },
	};
	return nvtxRangeStartEx(&eventAttrib);
}

static inline void nvtx_end(nvtxRangeId_t id) {
	nvtxRangeEnd(id);
}
#else
static inline nvtxRangeId_t nvtx_start(const char* name){ (void)name; return 0 }
static inline void nvtx_end(nvtxRangeId_t id){(void)id;}
#endif
