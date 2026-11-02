/* Kokoro ups.1 polyphase -> interleaved + ReflectionPad1d((1,0)), fp16 as raw 16-bit, into B's
   x_after buffer. src: (6, 128, Q) plain; dst: plain (128, Lb) if c2 == 1, else NC1HWC2 (16, Lb, 8).
   Positions past 1 + 6Q are zeroed. */
#include <stdint.h>
#include <string.h>

void interleave(const uint16_t *src, uint16_t *dst, int Q, int Lb, int nc1hwc2)
{
    const long n = 6L * Q;
    if (!nc1hwc2) {
        for (int c = 0; c < 128; c++) {
            uint16_t *d = dst + (long)c * Lb;
            const uint16_t *s0 = src + (long)c * Q;               /* phase p at s0 + p*128*Q */
            const long ps = 128L * Q;
            d[0] = s0[ps];                                         /* reflection: y[1] = phase 1, q 0 */
            for (int q = 0; q < Q; q++) {
                uint16_t *o = d + 1 + 6L * q;
                o[0] = s0[q]; o[1] = s0[ps + q]; o[2] = s0[2 * ps + q];
                o[3] = s0[3 * ps + q]; o[4] = s0[4 * ps + q]; o[5] = s0[5 * ps + q];
            }
            memset(d + 1 + n, 0, (Lb - 1 - n) * 2);
        }
    } else {
        for (int c1 = 0; c1 < 16; c1++) {
            uint16_t *d = dst + (long)c1 * Lb * 8;
            const uint16_t *s[6][8];
            for (int p = 0; p < 6; p++)
                for (int c2 = 0; c2 < 8; c2++)
                    s[p][c2] = src + ((long)p * 128 + c1 * 8 + c2) * Q;
            for (int c2 = 0; c2 < 8; c2++) d[c2] = s[1][c2][0];
            for (int q = 0; q < Q; q++) {
                uint16_t *o = d + (1 + 6L * q) * 8;
                for (int p = 0; p < 6; p++)
                    for (int c2 = 0; c2 < 8; c2++)
                        o[p * 8 + c2] = s[p][c2][q];
            }
            memset(d + (1 + n) * 8, 0, (Lb - 1 - n) * 16);
        }
    }
}
