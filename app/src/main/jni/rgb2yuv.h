//
// Created by kewang on 4/18/17.
//

#ifndef DANCEDANCECONVOLUTION_RGB2YUV_H
#define DANCEDANCECONVOLUTION_RGB2YUV_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void ConvertARGB8888ToYUV420SP(const uint32_t* const input,
                               uint8_t* const output, int width, int height);

void ConvertRGB565ToYUV420SP(const uint16_t* const input, uint8_t* const output,
                             const int width, const int height);

#ifdef __cplusplus
}
#endif

#endif //DANCEDANCECONVOLUTION_RGB2YUV_H
