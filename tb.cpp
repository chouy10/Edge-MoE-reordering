// tb_ssmu_edge.cpp  (FIXED: TB contains ONLY reference + main; NO DUT function bodies)

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <vector>

#include "ssmu.h"
#include <hls_stream.h>
#include <hls_math.h>

// =============================================================
// IMPORTANT:
// - DO NOT define any DUT functions here (conv1d_silu_stream, projection_streams, ... SSMU)
// - DUT is compiled from ssmuedge.cpp (syn.file)
// - This TB only calls SSMU() declared in ssmu.h
// =============================================================

// local floating type for accurate math
typedef float FDTYPE;

// ---- TB-local elementwise ops (make them static to avoid any symbol conflict) ----
static inline DTYPE tb_silu(DTYPE a) {
    FDTYPE x = (FDTYPE)a;
    FDTYPE expv = hls::exp(-x);
    FDTYPE sig  = (FDTYPE)1.0 / ((FDTYPE)1.0 + expv);
    FDTYPE res  = x * sig;
    return (DTYPE)res;
}

static inline DTYPE tb_exp(DTYPE a) {
    FDTYPE x = (FDTYPE)a;
    FDTYPE y = hls::exp(x);
    return (DTYPE)y;
}

static inline DTYPE tb_softplus(DTYPE a) {
    FDTYPE x = (FDTYPE)a;
    FDTYPE y = hls::log((FDTYPE)1.0 + hls::exp(x));
    return (DTYPE)y;
}

// utilities
static inline float to_float(DTYPE x) { return (float)x; }
static inline DTYPE from_float(float x) { return (DTYPE)x; }

// deterministic RNG
static unsigned g_seed = 1;
static inline unsigned xorshift32() {
    unsigned x = g_seed;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    g_seed = x;
    return x;
}
static inline float frand(float lo, float hi) {
    unsigned r = xorshift32();
    float u = (r & 0xFFFFFF) / (float)0x1000000; // [0,1)
    return lo + (hi - lo) * u;
}

// Reference outputs
struct RefOut {
    std::vector<DTYPE_VEC> H1_out; // N*VEC_D
    std::vector<DTYPE_VEC> out;    // VEC_D
};

// Reference model mirroring your current DUT behavior
static RefOut run_reference(
    const std::vector<DTYPE> &kernel,                 // K
    const std::vector<DTYPE_VEC> &A_in,               // N
    DTYPE_VEC (*W_B)[VEC_D],                          // [N][VEC_D]
    DTYPE_VEC (*W_C)[VEC_D],                          // [N][VEC_D]
    DTYPE_VEC (*W_delta)[VEC_D],                      // [VEC_D][VEC_D]
    const std::vector<DTYPE_VEC> &X_in,               // VEC_D
    const std::vector<DTYPE_VEC> &H0_in               // N*VEC_D
) {
    // ---- Part 1: X_gate = silu(X) ----
    DTYPE kernel_buffer[K];
    for (int i = 0; i < K; ++i) kernel_buffer[i] = kernel[i];

    std::vector<DTYPE_VEC> X_gate(VEC_D);
    for (int i = 0; i < VEC_D; ++i) {
        DTYPE_VEC gate;
        for (int l = 0; l < VEC_FACTOR; ++l) gate[l] = tb_silu(X_in[i][l]);
        X_gate[i] = gate;
    }

    // ---- Part 1: X_ssm = silu(conv1d(X, kernel)) with same line_buffer behavior ----
    DTYPE line_buffer[K-1][VEC_FACTOR];
    for (int i = 0; i < K-1; ++i)
        for (int l = 0; l < VEC_FACTOR; ++l)
            line_buffer[i][l] = 0;

    std::vector<DTYPE_VEC> X_ssm(VEC_D);
    for (int i = 0; i < VEC_D; ++i) {
        DTYPE_VEC in_vec = X_in[i];

        DTYPE window[K][VEC_FACTOR];
        for (int j = 0; j < K-1; ++j)
            for (int l = 0; l < VEC_FACTOR; ++l)
                window[j][l] = line_buffer[j][l];
        for (int l = 0; l < VEC_FACTOR; ++l)
            window[K-1][l] = in_vec[l];

        for (int j = K-2; j > 0; --j)
            for (int l = 0; l < VEC_FACTOR; ++l)
                line_buffer[j][l] = line_buffer[j-1][l];
        for (int l = 0; l < VEC_FACTOR; ++l)
            line_buffer[0][l] = in_vec[l];

        DTYPE_VEC outv;
        for (int lane = 0; lane < VEC_FACTOR; ++lane) {
            FDTYPE sum = 0.0f;
            for (int kk = 0; kk < K; ++kk) {
                sum += (FDTYPE)kernel_buffer[kk] * (FDTYPE)window[kk][lane];
            }
            outv[lane] = tb_silu((DTYPE)sum);
        }
        X_ssm[i] = outv;
    }

    // ---- Part 2: delta ----
    std::vector<DTYPE_VEC> delta(VEC_D);
    for (int i = 0; i < VEC_D; ++i) {
        DTYPE_VEC acc;
        for (int l = 0; l < VEC_FACTOR; ++l) acc[l] = 0;

        for (int j = 0; j < VEC_D; ++j) {
            DTYPE_VEC xj = X_ssm[j];
            DTYPE_VEC w  = W_delta[i][j];
            for (int l = 0; l < VEC_FACTOR; ++l) {
                FDTYPE prod = (FDTYPE)xj[l] * (FDTYPE)w[l];
                acc[l] = (DTYPE)((FDTYPE)acc[l] + prod);
            }
        }
        DTYPE_VEC dv;
        for (int l = 0; l < VEC_FACTOR; ++l) dv[l] = tb_softplus(acc[l]);
        delta[i] = dv;
    }

    // ---- Part 2: B ----
    std::vector<DTYPE_VEC> B(N);
    for (int i = 0; i < N; ++i) {
        DTYPE_VEC outB;
        for (int l = 0; l < VEC_FACTOR; ++l) outB[l] = 0;
        for (int j = 0; j < VEC_D; ++j) {
            DTYPE_VEC xj = X_ssm[j];
            DTYPE_VEC w  = W_B[i][j];
            for (int l = 0; l < VEC_FACTOR; ++l) {
                outB[l] = (DTYPE)((FDTYPE)outB[l] + (FDTYPE)xj[l] * (FDTYPE)w[l]);
            }
        }
        B[i] = outB;
    }

    // ---- Part 2: C stream (N*VEC_D blocks; same as your current code) ----
    std::vector<DTYPE_VEC> C_stream;
    C_stream.reserve(N * VEC_D);
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < VEC_D; ++j) {
            (void)j; // note: j does not affect compute in your current C loop
            DTYPE_VEC outCij;
            for (int l = 0; l < VEC_FACTOR; ++l) outCij[l] = 0;
            for (int t = 0; t < VEC_D; ++t) {
                DTYPE_VEC xt = X_ssm[t];
                DTYPE_VEC w  = W_C[i][t];
                for (int l = 0; l < VEC_FACTOR; ++l) {
                    outCij[l] = (DTYPE)((FDTYPE)outCij[l] + (FDTYPE)xt[l] * (FDTYPE)w[l]);
                }
            }
            C_stream.push_back(outCij);
        }
    }

    // ---- Part 3: ddA (i,j) ----
    std::vector<DTYPE_VEC> ddA_stream;
    ddA_stream.reserve(N * VEC_D);
    for (int i = 0; i < N; ++i) {
        DTYPE_VEC A_vec = A_in[i];
        for (int j = 0; j < VEC_D; ++j) {
            DTYPE_VEC ddA;
            DTYPE_VEC dlt = delta[j];
            for (int l = 0; l < VEC_FACTOR; ++l) {
                DTYPE dA = (DTYPE)((FDTYPE)A_vec[l] * (FDTYPE)dlt[l]);
                ddA[l] = tb_exp(dA);
            }
            ddA_stream.push_back(ddA);
        }
    }

    // ---- Part 3: dB (i,j) ----
    std::vector<DTYPE_VEC> dB_stream;
    dB_stream.reserve(N * VEC_D);
    for (int i = 0; i < N; ++i) {
        DTYPE_VEC Bv = B[i];
        for (int j = 0; j < VEC_D; ++j) {
            DTYPE_VEC out;
            DTYPE_VEC dlt = delta[j];
            for (int l = 0; l < VEC_FACTOR; ++l) {
                out[l] = (DTYPE)((FDTYPE)Bv[l] * (FDTYPE)dlt[l]);
            }
            dB_stream.push_back(out);
        }
    }

    // ---- Part 4: H1 ----
    std::vector<DTYPE_VEC> H1_stream;
    H1_stream.reserve(N * VEC_D);
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < VEC_D; ++j) {
            DTYPE_VEC H0v = H0_in[i * VEC_D + j];
            DTYPE_VEC ddA = ddA_stream[i * VEC_D + j];
            DTYPE_VEC dBv = dB_stream[i * VEC_D + j];
            DTYPE_VEC dx  = X_ssm[j];

            DTYPE_VEC H1v;
            for (int l = 0; l < VEC_FACTOR; ++l) {
                FDTYPE v = (FDTYPE)H0v[l] * (FDTYPE)ddA[l]
                         + (FDTYPE)dBv[l] * (FDTYPE)dx[l];
                H1v[l] = (DTYPE)v;
            }
            H1_stream.push_back(H1v);
        }
    }

    // ---- Part 5: final out ----
    DTYPE_VEC acc[VEC_D];
    for (int j = 0; j < VEC_D; ++j)
        for (int l = 0; l < VEC_FACTOR; ++l)
            acc[j][l] = 0;

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < VEC_D; ++j) {
            DTYPE_VEC H1v = H1_stream[i * VEC_D + j];
            DTYPE_VEC Cij = C_stream[i * VEC_D + j];
            for (int l = 0; l < VEC_FACTOR; ++l) {
                acc[j][l] = (DTYPE)((FDTYPE)acc[j][l] + (FDTYPE)H1v[l] * (FDTYPE)Cij[l]);
            }
        }
    }

    std::vector<DTYPE_VEC> out_vec(VEC_D);
    for (int j = 0; j < VEC_D; ++j) {
        DTYPE_VEC ov;
        for (int l = 0; l < VEC_FACTOR; ++l) {
            ov[l] = (DTYPE)((FDTYPE)X_gate[j][l] + (FDTYPE)acc[j][l]);
        }
        out_vec[j] = ov;
    }

    RefOut R;
    R.H1_out = std::move(H1_stream);
    R.out    = std::move(out_vec);
    return R;
}

// =============================================================
// Main TB
// =============================================================
int main() {
    g_seed = 1;

    // 1) Random inputs (small range to avoid overflow)
    std::vector<DTYPE> kernel(K);
    for (int i = 0; i < K; ++i) kernel[i] = from_float(frand(-0.25f, 0.25f));

    std::vector<DTYPE_VEC> X_in(VEC_D);
    for (int j = 0; j < VEC_D; ++j) {
        DTYPE_VEC v;
        for (int l = 0; l < VEC_FACTOR; ++l) v[l] = from_float(frand(-0.5f, 0.5f));
        X_in[j] = v;
    }

    std::vector<DTYPE_VEC> A_in(N);
    for (int i = 0; i < N; ++i) {
        DTYPE_VEC v;
        for (int l = 0; l < VEC_FACTOR; ++l) v[l] = from_float(frand(-0.25f, 0.25f));
        A_in[i] = v;
    }

    std::vector<DTYPE_VEC> H0_in(N * VEC_D);
    for (int idx = 0; idx < N * VEC_D; ++idx) {
        DTYPE_VEC v;
        for (int l = 0; l < VEC_FACTOR; ++l) v[l] = from_float(frand(-0.25f, 0.25f));
        H0_in[idx] = v;
    }

    static DTYPE_VEC W_B[N][VEC_D];
    static DTYPE_VEC W_C[N][VEC_D];
    static DTYPE_VEC W_delta[VEC_D][VEC_D];

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < VEC_D; ++j) {
            DTYPE_VEC wb, wc;
            for (int l = 0; l < VEC_FACTOR; ++l) {
                wb[l] = from_float(frand(-0.125f, 0.125f));
                wc[l] = from_float(frand(-0.125f, 0.125f));
            }
            W_B[i][j] = wb;
            W_C[i][j] = wc;
        }
    }

    for (int i = 0; i < VEC_D; ++i) {
        for (int j = 0; j < VEC_D; ++j) {
            DTYPE_VEC wd;
            for (int l = 0; l < VEC_FACTOR; ++l) wd[l] = from_float(frand(-0.125f, 0.125f));
            W_delta[i][j] = wd;
        }
    }

    // 2) Run reference
    RefOut ref = run_reference(kernel, A_in, &W_B[0], &W_C[0], &W_delta[0], X_in, H0_in);

    // 3) Prepare DUT streams
    hls::stream<DTYPE>      kernel_s;
    hls::stream<DTYPE_VEC>  A_s;
    hls::stream<DTYPE_VEC>  X_s;
    hls::stream<DTYPE_VEC>  H0_s;
    hls::stream<DTYPE_VEC>  H1_out_s;
    hls::stream<DTYPE_VEC>  out_s;

    // push inputs
    for (int i = 0; i < K; ++i) kernel_s.write(kernel[i]);
    for (int i = 0; i < N; ++i) A_s.write(A_in[i]);
    for (int j = 0; j < VEC_D; ++j) X_s.write(X_in[j]);
    for (int idx = 0; idx < N * VEC_D; ++idx) H0_s.write(H0_in[idx]);

    // 4) Run DUT (implementation is in ssmuedge.cpp)
    SSMU(kernel_s, A_s, W_B, W_C, W_delta, X_s, H0_s, H1_out_s, out_s);

    // 5) Read outputs (with empty-check)
    std::vector<DTYPE_VEC> dut_H1(N * VEC_D);
    for (int idx = 0; idx < N * VEC_D; ++idx) {
        if (H1_out_s.empty()) {
            std::printf("ERROR: H1_out_s empty at idx=%d (expected %d items)\n", idx, N * VEC_D);
            return 3;
        }
        dut_H1[idx] = H1_out_s.read();
    }

    std::vector<DTYPE_VEC> dut_out(VEC_D);
    for (int j = 0; j < VEC_D; ++j) {
        if (out_s.empty()) {
            std::printf("ERROR: out_s empty at j=%d (expected %d items)\n", j, VEC_D);
            return 3;
        }
        dut_out[j] = out_s.read();
    }

    // 6) Compare
    // NOTE: LSB/TOL depends on your DTYPE(ap_fixed) frac bits.
    const float LSB = 1.0f / 4096.0f;  // example: ap_fixed<16,4> => frac=12
    const float TOL = 64.0f * LSB;

    int err = 0;

    auto cmp_vec = [&](const DTYPE_VEC &a, const DTYPE_VEC &b, const char *tag, int idx) {
        for (int l = 0; l < VEC_FACTOR; ++l) {
            float ra = to_float(a[l]);
            float rb = to_float(b[l]);
            float diff = std::fabs(ra - rb);
            if (diff > TOL) {
                if (err < 10) {
                    std::printf("Mismatch %s[%d][lane=%d]: ref=%f dut=%f diff=%f\n",
                                tag, idx, l, ra, rb, diff);
                }
                err++;
            }
        }
    };

    for (int idx = 0; idx < N * VEC_D; ++idx) cmp_vec(ref.H1_out[idx], dut_H1[idx], "H1", idx);
    for (int j = 0; j < VEC_D; ++j)          cmp_vec(ref.out[j],    dut_out[j], "OUT", j);

    if (err == 0) {
        std::printf("PASS (all outputs match within tol=%g)\n", TOL);
        return 0;
    } else {
        std::printf("FAIL total mismatches=%d (showing up to 10)\n", err);
        return 1;
    }
}
