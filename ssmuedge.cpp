#include "ssmu.h"
// #include "lut_optimized.h"

// local floating type for accurate math (change to ap_fixed if desired)
typedef float FDTYPE;

// compute sl = x * sigmoid(x)
static inline DTYPE silu_elem(DTYPE a) {
    FDTYPE x = (FDTYPE)a;
    FDTYPE expv = hls::exp(-x);
    FDTYPE sig  = (FDTYPE)1.0 / ((FDTYPE)1.0 + expv);
    FDTYPE res  = x * sig;
    return (DTYPE)res;
}

static inline DTYPE exp_elem(DTYPE a) {
    FDTYPE x = (FDTYPE)a;
    FDTYPE y = hls::exp(x);
    return (DTYPE)y;
}

static inline DTYPE softplus_elem(DTYPE a) {
    FDTYPE x = (FDTYPE)a;
    FDTYPE y = hls::log((FDTYPE)1.0 + hls::exp(x));
    return (DTYPE)y;
}

// ==================== Part 1: X -> X_gate, X_ssm ====================
void conv1d_silu_stream(hls::stream<DTYPE_VEC>& X_in,
                        hls::stream<DTYPE>& kernel_in,
                        hls::stream<DTYPE_VEC>& X_gate_out,
                        hls::stream<DTYPE_VEC>& X_ssm_out) {
#pragma HLS INLINE off

    static DTYPE line_buffer[K-1][VEC_FACTOR];
#pragma HLS ARRAY_PARTITION variable=line_buffer complete dim=2
#pragma HLS BIND_STORAGE variable=line_buffer type=ram_s2p impl=bram

    DTYPE kernel_buffer[K];
#pragma HLS ARRAY_PARTITION variable=kernel_buffer complete
    for (int i = 0; i < K; ++i) {
        kernel_buffer[i] = kernel_in.read();
    }

    DTYPE_VEC X_buffer[VEC_D];
#pragma HLS BIND_STORAGE variable=X_buffer type=ram_s2p impl=bram

read_and_gate:
    for (int i = 0; i < VEC_D; ++i) {
#pragma HLS PIPELINE II=1
        DTYPE_VEC xv = X_in.read();
        X_buffer[i] = xv;

        DTYPE_VEC gate_out;
        for (int k = 0; k < VEC_FACTOR; ++k) {
#pragma HLS UNROLL factor=4
            gate_out[k] = silu_elem(xv[k]);
        }
        X_gate_out.write(gate_out);
    }

    // init line buffer
    for (int i = 0; i < K-1; ++i) {
        for (int k = 0; k < VEC_FACTOR; ++k) {
            line_buffer[i][k] = 0;
        }
    }

conv_proc:
    for (int i = 0; i < VEC_D; ++i) {
#pragma HLS PIPELINE II=2
        DTYPE_VEC in_vec = X_buffer[i];

        DTYPE window[K][VEC_FACTOR];
#pragma HLS ARRAY_PARTITION variable=window complete dim=2

        for (int j = 0; j < K-1; ++j) {
            for (int k = 0; k < VEC_FACTOR; ++k) {
                window[j][k] = line_buffer[j][k];
            }
        }
        for (int k = 0; k < VEC_FACTOR; ++k) {
            window[K-1][k] = in_vec[k];
        }

        // shift
        for (int j = K-2; j > 0; --j) {
            for (int k = 0; k < VEC_FACTOR; ++k) {
                line_buffer[j][k] = line_buffer[j-1][k];
            }
        }
        for (int k = 0; k < VEC_FACTOR; ++k) {
            line_buffer[0][k] = in_vec[k];
        }

        // conv
        DTYPE_VEC conv_out;
        for (int lane = 0; lane < VEC_FACTOR; ++lane) {
#pragma HLS UNROLL factor=4
            FDTYPE sum = 0.0f;
            for (int kk = 0; kk < K; ++kk) {
                sum += (FDTYPE)kernel_buffer[kk] * (FDTYPE)window[kk][lane];
            }
            conv_out[lane] = (DTYPE)sum;
        }

        // silu(conv)
        DTYPE_VEC ssm_out;
        for (int k = 0; k < VEC_FACTOR; ++k) {
#pragma HLS UNROLL factor=4
            ssm_out[k] = silu_elem(conv_out[k]);
        }
        X_ssm_out.write(ssm_out);
    }
}

// ==================== Part 2: projections ====================
void projection_streams(hls::stream<DTYPE_VEC>& X_ssm_in,
                        DTYPE_VEC W_B[N][VEC_D],
                        DTYPE_VEC W_C[N][VEC_D],
                        DTYPE_VEC W_delta[VEC_D][VEC_D],
                        hls::stream<DTYPE_VEC>& B_out,
                        hls::stream<DTYPE_VEC>& C_out,
                        hls::stream<DTYPE_VEC>& delta_out_A,
                        hls::stream<DTYPE_VEC>& delta_out_B) {
#pragma HLS INLINE off

    DTYPE_VEC X_buf[VEC_D];
#pragma HLS ARRAY_PARTITION variable=X_buf cyclic factor=2
#pragma HLS BIND_STORAGE variable=X_buf type=ram_s2p impl=bram

#pragma HLS BIND_STORAGE variable=W_B type=ram_s2p impl=bram
#pragma HLS BIND_STORAGE variable=W_C type=ram_s2p impl=bram
#pragma HLS BIND_STORAGE variable=W_delta type=ram_s2p impl=bram

    for (int j = 0; j < VEC_D; ++j) {
#pragma HLS PIPELINE II=1
        X_buf[j] = X_ssm_in.read();
    }

compute_delta:
    for (int i = 0; i < VEC_D; ++i) {
#pragma HLS PIPELINE II=4
        DTYPE_VEC acc;
        for (int l = 0; l < VEC_FACTOR; ++l) acc[l] = 0;

        for (int j = 0; j < VEC_D; ++j) {
            DTYPE_VEC xj = X_buf[j];
            DTYPE_VEC w  = W_delta[i][j];
            for (int l = 0; l < VEC_FACTOR; ++l) {
#pragma HLS UNROLL factor=4
                FDTYPE prod = (FDTYPE)xj[l] * (FDTYPE)w[l];
                acc[l] = (DTYPE)((FDTYPE)acc[l] + prod);
            }
        }

        DTYPE_VEC delta_vec;
        for (int l = 0; l < VEC_FACTOR; ++l) {
#pragma HLS UNROLL factor=4
            delta_vec[l] = softplus_elem(acc[l]);
        }

        delta_out_A.write(delta_vec);
        delta_out_B.write(delta_vec);
    }

    // B
    for (int i = 0; i < N; ++i) {
#pragma HLS PIPELINE II=4
        DTYPE_VEC outB;
        for (int l = 0; l < VEC_FACTOR; ++l) outB[l] = (DTYPE)0;

        for (int j = 0; j < VEC_D; ++j) {
            DTYPE_VEC xj = X_buf[j];
            DTYPE_VEC w  = W_B[i][j];
            for (int l = 0; l < VEC_FACTOR; ++l) {
#pragma HLS UNROLL
                outB[l] = (DTYPE)((FDTYPE)outB[l] + (FDTYPE)xj[l] * (FDTYPE)w[l]);
            }
        }
        B_out.write(outB);
    }

    // C 
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < VEC_D; ++j) {
#pragma HLS PIPELINE II=4
            DTYPE_VEC outCij;
            for (int l = 0; l < VEC_FACTOR; ++l) outCij[l] = (DTYPE)0;

            for (int t = 0; t < VEC_D; ++t) {
                DTYPE_VEC x_t = X_buf[t];
                DTYPE_VEC w   = W_C[i][t];
                for (int l = 0; l < VEC_FACTOR; ++l) {
#pragma HLS UNROLL
                    outCij[l] = (DTYPE)((FDTYPE)outCij[l] + (FDTYPE)x_t[l] * (FDTYPE)w[l]);
                }
            }
            C_out.write(outCij);
        }
    }
}

// ==================== Part 3: A -> ddA ====================
void A_to_ddA_stream(hls::stream<DTYPE_VEC>& A_in,
                     hls::stream<DTYPE_VEC>& delta_in,
                     hls::stream<DTYPE_VEC>& ddA_out) {
#pragma HLS INLINE off

    DTYPE_VEC delta_buf[VEC_D];
#pragma HLS BIND_STORAGE variable=delta_buf type=ram_s2p impl=bram
    for (int j = 0; j < VEC_D; ++j) {
#pragma HLS PIPELINE II=1
        delta_buf[j] = delta_in.read();
    }

    for (int i = 0; i < N; ++i) {
#pragma HLS PIPELINE II=4
        DTYPE_VEC A_vec = A_in.read();

        for (int j = 0; j < VEC_D; ++j) {
            DTYPE_VEC ddA;
            DTYPE_VEC dlt = delta_buf[j];
            for (int l = 0; l < VEC_FACTOR; ++l) {
#pragma HLS UNROLL factor=4
                DTYPE dA = (DTYPE)((FDTYPE)A_vec[l] * (FDTYPE)dlt[l]);
                ddA[l] = exp_elem(dA);
            }
            ddA_out.write(ddA);
        }
    }
}

// ==================== Part 3b: B -> dB ====================
void B_to_dB_stream(hls::stream<DTYPE_VEC>& B_in,
                    hls::stream<DTYPE_VEC>& delta_in,
                    hls::stream<DTYPE_VEC>& dB_out) {
#pragma HLS INLINE off

    DTYPE_VEC delta_buf[VEC_D];
#pragma HLS BIND_STORAGE variable=delta_buf type=ram_s2p impl=bram
    for (int j = 0; j < VEC_D; ++j) {
#pragma HLS PIPELINE II=1
        delta_buf[j] = delta_in.read();
    }

    for (int i = 0; i < N; ++i) {
#pragma HLS PIPELINE II=4
        DTYPE_VEC Bv = B_in.read();

        for (int j = 0; j < VEC_D; ++j) {
            DTYPE_VEC out;
            DTYPE_VEC dlt = delta_buf[j];
            for (int l = 0; l < VEC_FACTOR; ++l) {
#pragma HLS UNROLL factor=4
                out[l] = (DTYPE)((FDTYPE)Bv[l] * (FDTYPE)dlt[l]);
            }
            dB_out.write(out);
        }
    }
}

// ==================== Part 4: Update H ====================
void update_H_stream(hls::stream<DTYPE_VEC>& ddA_in,
                     hls::stream<DTYPE_VEC>& dX_in,
                     hls::stream<DTYPE_VEC>& dB_in,
                     hls::stream<DTYPE_VEC>& H0_in,
                     hls::stream<DTYPE_VEC>& H1_out) {
#pragma HLS INLINE off

    DTYPE_VEC dX_buf[VEC_D];
#pragma HLS ARRAY_PARTITION variable=dX_buf cyclic factor=2
    for (int j = 0; j < VEC_D; ++j) {
#pragma HLS PIPELINE II=1
        dX_buf[j] = dX_in.read();
    }

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < VEC_D; ++j) {
#pragma HLS PIPELINE II=4
            DTYPE_VEC H0v = H0_in.read();
            DTYPE_VEC ddA = ddA_in.read();
            DTYPE_VEC dBv = dB_in.read();
            DTYPE_VEC dx  = dX_buf[j];

            DTYPE_VEC H1v;
            for (int l = 0; l < VEC_FACTOR; ++l) {
#pragma HLS UNROLL
                FDTYPE v = (FDTYPE)H0v[l] * (FDTYPE)ddA[l]
                         + (FDTYPE)dBv[l] * (FDTYPE)dx[l];
                H1v[l] = (DTYPE)v;
            }
            H1_out.write(H1v);
        }
    }
}

// ==================== duplicators ====================
void duplicate_H1_stream(hls::stream<DTYPE_VEC>& in,
                         hls::stream<DTYPE_VEC>& out1,
                         hls::stream<DTYPE_VEC>& out2) {
#pragma HLS INLINE off
    for (int i = 0; i < N * VEC_D; ++i) {
#pragma HLS PIPELINE II=1
        DTYPE_VEC v = in.read();
        out1.write(v);
        out2.write(v);
    }
}

void dup_vecD_stream(hls::stream<DTYPE_VEC>& in,
                     hls::stream<DTYPE_VEC>& out1,
                     hls::stream<DTYPE_VEC>& out2) {
#pragma HLS INLINE off
    for (int i = 0; i < VEC_D; ++i) {
#pragma HLS PIPELINE II=1
        DTYPE_VEC v = in.read();
        out1.write(v);
        out2.write(v);
    }
}

// ==================== Part 5: final output ====================
void final_output_stream_edgemoE(hls::stream<DTYPE_VEC>& X_gate_in,
                                 hls::stream<DTYPE_VEC>& H1_in,
                                 hls::stream<DTYPE_VEC>& C_in,
                                 hls::stream<DTYPE_VEC>& out) {
#pragma HLS INLINE off

    DTYPE_VEC X_gate[VEC_D];
#pragma HLS ARRAY_PARTITION variable=X_gate cyclic factor=2
    for (int j = 0; j < VEC_D; ++j) {
#pragma HLS PIPELINE II=1
        X_gate[j] = X_gate_in.read();
    }

    DTYPE_VEC acc[VEC_D];
#pragma HLS ARRAY_PARTITION variable=acc cyclic factor=2
    for (int j = 0; j < VEC_D; ++j) {
        for (int l = 0; l < VEC_FACTOR; ++l) {
#pragma HLS UNROLL
            acc[j][l] = (DTYPE)0;
        }
    }

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < VEC_D; ++j) {
#pragma HLS PIPELINE II=4
            DTYPE_VEC H1v = H1_in.read();
            DTYPE_VEC Cij = C_in.read();
            for (int l = 0; l < VEC_FACTOR; ++l) {
#pragma HLS UNROLL
                acc[j][l] = (DTYPE)((FDTYPE)acc[j][l] + (FDTYPE)H1v[l] * (FDTYPE)Cij[l]);
            }
        }
    }

    for (int j = 0; j < VEC_D; ++j) {
#pragma HLS PIPELINE II=1
        DTYPE_VEC outv;
        for (int l = 0; l < VEC_FACTOR; ++l) {
#pragma HLS UNROLL
            outv[l] = (DTYPE)((FDTYPE)X_gate[j][l] + (FDTYPE)acc[j][l]);
        }
        out.write(outv);
    }
}

// ==================== TOP ====================
void SSMU(
    hls::stream<DTYPE>& kernel_in,
    hls::stream<DTYPE_VEC>& A_in,
    DTYPE_VEC W_B[N][VEC_D],
    DTYPE_VEC W_C[N][VEC_D],
    DTYPE_VEC W_delta[VEC_D][VEC_D],
    hls::stream<DTYPE_VEC>& X_in,
    hls::stream<DTYPE_VEC>& H0_in,
    hls::stream<DTYPE_VEC>& H1_out,
    hls::stream<DTYPE_VEC>& out) {

#pragma HLS DATAFLOW

    hls::stream<DTYPE_VEC> X_gate_stream("X_gate_stream");
    hls::stream<DTYPE_VEC> X_ssm_stream ("X_ssm_stream");

    hls::stream<DTYPE_VEC> X_ssm_proj_stream("X_ssm_proj_stream");
    hls::stream<DTYPE_VEC> X_ssm_upd_stream ("X_ssm_upd_stream");

    hls::stream<DTYPE_VEC> B_stream("B_stream");
    hls::stream<DTYPE_VEC> C_stream("C_stream");
    hls::stream<DTYPE_VEC> delta_stream_A("delta_stream_A");
    hls::stream<DTYPE_VEC> delta_stream_B("delta_stream_B");
    hls::stream<DTYPE_VEC> ddA_stream("ddA_stream");
    hls::stream<DTYPE_VEC> dB_stream("dB_stream");
    hls::stream<DTYPE_VEC> H1_temp_stream("H1_temp_stream");
    hls::stream<DTYPE_VEC> H1_final_stream("H1_final_stream");

#pragma HLS STREAM variable=X_gate_stream      depth=4
#pragma HLS STREAM variable=X_ssm_stream       depth=4
#pragma HLS STREAM variable=X_ssm_proj_stream  depth=4
#pragma HLS STREAM variable=X_ssm_upd_stream   depth=4
#pragma HLS STREAM variable=B_stream           depth=4
#pragma HLS STREAM variable=C_stream           depth=4
#pragma HLS STREAM variable=delta_stream_A     depth=4
#pragma HLS STREAM variable=delta_stream_B     depth=4
#pragma HLS STREAM variable=ddA_stream         depth=4
#pragma HLS STREAM variable=dB_stream          depth=4
#pragma HLS STREAM variable=H1_temp_stream     depth=4
#pragma HLS STREAM variable=H1_final_stream    depth=4

    conv1d_silu_stream(X_in, kernel_in, X_gate_stream, X_ssm_stream);

    // critical fix: split X_ssm to avoid double-consume
    dup_vecD_stream(X_ssm_stream, X_ssm_proj_stream, X_ssm_upd_stream);

    projection_streams(X_ssm_proj_stream, W_B, W_C, W_delta,
                       B_stream, C_stream, delta_stream_A, delta_stream_B);

    A_to_ddA_stream(A_in, delta_stream_A, ddA_stream);
    B_to_dB_stream(B_stream, delta_stream_B, dB_stream);

    update_H_stream(ddA_stream, X_ssm_upd_stream, dB_stream, H0_in, H1_temp_stream);

    duplicate_H1_stream(H1_temp_stream, H1_final_stream, H1_out);

    final_output_stream_edgemoE(X_gate_stream, H1_final_stream, C_stream, out);
}
