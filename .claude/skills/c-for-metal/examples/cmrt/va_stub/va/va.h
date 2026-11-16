// va/va.h (stub) -- the few libva declarations the CM runtime headers need.
// Only for EMULATION builds of the CM runtime on machines without the
// libva development headers (Debian/Ubuntu: `apt install libva-dev`); prefer
// the real header when it is installed. The values match libva's va.h.
#pragma once
#include <stdint.h>

typedef void *VADisplay;
typedef unsigned int VASurfaceID;

#define VA_FOURCC(ch0, ch1, ch2, ch3)                                          \
  ((unsigned long)(unsigned char)(ch0) | ((unsigned long)(unsigned char)(ch1) << 8) | \
   ((unsigned long)(unsigned char)(ch2) << 16) | ((unsigned long)(unsigned char)(ch3) << 24))

#define VA_FOURCC_411P VA_FOURCC('4', '1', '1', 'P')
#define VA_FOURCC_411R VA_FOURCC('4', '1', '1', 'R')
#define VA_FOURCC_422H VA_FOURCC('4', '2', '2', 'H')
#define VA_FOURCC_422V VA_FOURCC('4', '2', '2', 'V')
#define VA_FOURCC_444P VA_FOURCC('4', '4', '4', 'P')
#define VA_FOURCC_AI44 VA_FOURCC('A', 'I', '4', '4')
#define VA_FOURCC_AYUV VA_FOURCC('A', 'Y', 'U', 'V')
#define VA_FOURCC_BGRP VA_FOURCC('B', 'G', 'R', 'P')
#define VA_FOURCC_IMC3 VA_FOURCC('I', 'M', 'C', '3')
#define VA_FOURCC_NV12 VA_FOURCC('N', 'V', '1', '2')
#define VA_FOURCC_P010 VA_FOURCC('P', '0', '1', '0')
#define VA_FOURCC_P016 VA_FOURCC('P', '0', '1', '6')
#define VA_FOURCC_P208 VA_FOURCC('P', '2', '0', '8')
#define VA_FOURCC_RGBP VA_FOURCC('R', 'G', 'B', 'P')
#define VA_FOURCC_UYVY VA_FOURCC('U', 'Y', 'V', 'Y')
#define VA_FOURCC_Y210 VA_FOURCC('Y', '2', '1', '0')
#define VA_FOURCC_Y216 VA_FOURCC('Y', '2', '1', '6')
#define VA_FOURCC_Y410 VA_FOURCC('Y', '4', '1', '0')
#define VA_FOURCC_Y416 VA_FOURCC('Y', '4', '1', '6')
#define VA_FOURCC_YUY2 VA_FOURCC('Y', 'U', 'Y', '2')
#define VA_FOURCC_YV12 VA_FOURCC('Y', 'V', '1', '2')
