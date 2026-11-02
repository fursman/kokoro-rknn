#!/usr/bin/env python3
"""Whole-sentence NPU decoder graphs for one bucket length, from the reference model itself.

Source: models/dec.onnx, cut out of kokoro-v1.0.onnx by build/extract.py, so the weights
and ops are the reference's own. For a bucket of Tb frames this writes two fixed-shape
graphs, <out>_A.onnx and <out>_B.onnx, for rknn-toolkit2:

  * the graph ends at conv_post (spec/phase logits, 22 x (120T+1)); exp/sin and the
    model's own iSTFT run on the CPU (models/tail.onnx);
  * inputs are padded to Tb with zeros, and masks come in at every time resolution;
  * MASKED STATISTICS: every time-axis mean of the AdaIN / InstanceNorm layers becomes
    GlobalAveragePool(x * w), w = mask * (padded length / real length), so the
    statistics are those of the real frames only, as in the unpadded model;
  * every mixing op (Conv / ConvTranspose with kernel > 1) has its input multiplied by
    the binary mask, so the real frames see zeros past the end, as in the unpadded model;
  * Snake's sin^2 is RANGE-REDUCED (sin^2 has period pi): r = y - pi*round(y/pi), then a
    polynomial in r^2. The round is a Cast float -> int16 -> float, which runs on the
    NPU and rounds to nearest there (Round and Floor fall back to the CPU);
  * with --fuse (the default) each AdaIN becomes one per-channel affine x*A + B, with the
    Snake's alpha/pi folded into it and pi/alpha into the next conv's weights;
  * the 8191 WIDTH LIMIT: elementwise ops that broadcast over a time axis wider than 8191
    fail at compile time ("REGTASK: bit width of field value exceeds the limit") and the
    compiled model hangs the NPU. So the graph is split at ups.1: A runs at the frame
    rates and ends in a POLYPHASE form of the stride-6 ConvTranspose (one conv with 6 x 128
    output channels); the CPU interleaves the phases and adds the reflection pad; B runs
    the 120T+1 stage with every elementwise op on a (1, C, H, W) view (view4d below);
  * the stride-10 ConvTranspose (ups.0; librknnc only takes strides 1, 2, 4, 8) becomes
    zero-interleave + stride-1 Conv (exact).

The zero-interleave rewrite of the stride-10 ConvTranspose and the clipped-polynomial Sin
(`--sin clip`, kept only to reproduce the earlier NPU build in the evaluation) are adapted
from build.py in marty1885/kokoro-server, which also showed that GlobalAveragePool keeps
the normalisation on the NPU and that Snake's 1/alpha can be folded into the next conv.
Modified for this project; see NOTICE.

    python build/surgery.py --models models --T 128 --out models/buckets/dec_T128.onnx
    python build/surgery.py --models models --T 128 --sin rr --no-fuse --out check_T128.onnx   # fp32 check graph
"""
import argparse

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from onnx.utils import Extractor

X_POST = "/decoder/decoder/generator/conv_post/Conv_output_0"


def sin2_coeffs(deg, half=np.pi / 2):
    """Least-squares (Chebyshev nodes) fit of sin^2(r) = sum_k a_k u^k, u = r^2, |r| <= half."""
    t = np.cos(np.linspace(0, np.pi, 4001)) * 0.5 + 0.5
    r = t * half
    u = r * r
    A = np.stack([u ** k for k in range(1, deg + 1)], 1)
    a = np.linalg.lstsq(A, np.sin(r) ** 2, rcond=None)[0]
    rr = np.linspace(0, half, 100001)
    err = np.abs(sum(a[k - 1] * rr ** (2 * k) for k in range(1, deg + 1)) - np.sin(rr) ** 2).max()
    return a, err


def clip_sin_coeffs(degree=7, clip=3.14159):
    """The earlier NPU build's Sin (adapted from kokoro-server's build.py): an odd
    Chebyshev-fit polynomial on [-clip, clip], with the input clipped to that range.
    About 0.9% of Snake inputs have |alpha x| > pi (up to ~120), where this is wrong."""
    from numpy.polynomial import chebyshev
    t = np.cos(np.linspace(0, np.pi, 4096))
    cheb = chebyshev.Chebyshev.fit(t, np.sin(clip * t), degree)
    poly = cheb.convert(kind=np.polynomial.Polynomial).coef
    return [float(poly[p] / clip ** p) for p in (1, 3, 5, 7, 9) if p <= degree]


class Names:
    def __init__(self, g):
        self.g = g
        self.k = 0

    def const(self, val, dtype=np.float32):
        self.k += 1
        name = f"npu2_c{self.k}"
        self.g.initializer.append(numpy_helper.from_array(np.asarray(val, dtype), name))
        return name

    def tmp(self, base):
        self.k += 1
        return f"npu2/{base}_{self.k}"


def sin2_nodes(nm, y, out, mode, deg=4):
    """Nodes computing out = sin(y)^2."""
    nodes = []
    if mode == "cpu":                    # exact Sin (librknnrt 2.5.0 runs it on the CPU)
        s_ = nm.tmp("sin")
        nodes.append(helper.make_node("Sin", [y], [s_]))
        nodes.append(helper.make_node("Mul", [s_, s_], [out]))
    elif mode == "clip":                 # the original NPU build
        c = clip_sin_coeffs()
        xc, x2 = nm.tmp("xc"), nm.tmp("x2")
        nodes.append(helper.make_node("Clip", [y, nm.const(-3.14159), nm.const(3.14159)], [xc]))
        nodes.append(helper.make_node("Mul", [xc, xc], [x2]))
        acc = nm.const(c[-1])
        for ci in c[-2::-1]:
            t1, t2 = nm.tmp("hm"), nm.tmp("ha")
            nodes.append(helper.make_node("Mul", [x2, acc], [t1]))
            nodes.append(helper.make_node("Add", [t1, nm.const(ci)], [t2]))
            acc = t2
        s_ = nm.tmp("sin")
        nodes.append(helper.make_node("Mul", [xc, acc], [s_]))
        nodes.append(helper.make_node("Mul", [s_, s_], [out]))
    elif mode in ("rr", "rr_npu"):      # range-reduced: sin^2 is even with period pi
        # r = y - pi * round(y / pi), |r| <= pi/2. "rr" uses Round (exact semantics, for the
        # fp32 check on the CPU); "rr_npu" uses Cast float->int16->float, which librknnrt runs
        # on the NPU and which rounds to nearest there (measured), where Round/Floor fall
        # back to the CPU.
        a, _ = sin2_coeffs(deg)
        q, qr, kp, r, u = (nm.tmp(x) for x in ("q", "qr", "kp", "r", "u"))
        nodes.append(helper.make_node("Mul", [y, nm.const(1 / np.pi)], [q]))
        if mode == "rr":
            nodes.append(helper.make_node("Round", [q], [qr]))
        else:
            qi = nm.tmp("qi")
            nodes.append(helper.make_node("Cast", [q], [qi], to=TensorProto.INT16))
            nodes.append(helper.make_node("Cast", [qi], [qr], to=TensorProto.FLOAT))
        nodes.append(helper.make_node("Mul", [qr, nm.const(np.pi)], [kp]))
        nodes.append(helper.make_node("Sub", [y, kp], [r]))
        nodes.append(helper.make_node("Mul", [r, r], [u]))
        acc = nm.const(a[-1])
        for ai in a[-2::-1]:
            t1, t2 = nm.tmp("hm"), nm.tmp("ha")
            nodes.append(helper.make_node("Mul", [u, acc], [t1]))
            nodes.append(helper.make_node("Add", [t1, nm.const(ai)], [t2]))
            acc = t2
        nodes.append(helper.make_node("Mul", [u, acc], [out]))
    else:
        raise SystemExit(f"unknown sin mode {mode}")
    return nodes


def sin2_over_pi_coeffs(deg):
    """P(d) ~= sin^2(pi d) / pi = sum_k c_k u^k, u = d^2, |d| <= 1/2 (rescaled sin2_coeffs)."""
    a, err = sin2_coeffs(deg)
    return [a[k - 1] * np.pi ** (2 * k) / np.pi for k in range(1, deg + 1)], err


def fuse_norm_snake(g, nm, mask_norm, sin_mode, deg, w_of, var2=True):
    """Rewrite each AdaIN (and the Snake after it, when there is one) with as few full-size
    passes as possible. The reference computes, per channel over time,
        mu = mean(x); var = mean((x-mu)^2); n = (x-mu)/sqrt(var+eps); a = (1+g)*n + beta
        snake(a) = a + sin^2(alpha a)/alpha  -> next Conv
    Here (statistics over the valid frames, w = mask * Lb/L):
        mu = GAP(x w); m2 = GAP((x w) x); var = m2 - mu^2              (tiny tensors from here)
        A = (1+g)/sqrt(var+eps); B = beta - mu A
      no snake:  a = x A + B
      snake:     q = x (A alpha/pi) + B alpha/pi;  d = q - round(q);  z = q + P(d^2)
                 with P(d^2) = sin^2(pi d)/pi, and the next Conv's weights times pi/alpha
                 (per input channel), since snake(a) = (pi/alpha) z exactly.
    Returns the set of node ids removed and the new nodes, keyed by the node they replace."""
    nodes = list(g.node)
    by_name = {n.name: n for n in nodes}
    consumers = {}
    for n in nodes:
        for i in n.input:
            consumers.setdefault(i, []).append(n)
    init = {i.name: i for i in g.initializer}
    remove, insert_before = set(), {}
    n_ad = n_sn = 0
    for rm in [n for n in nodes if n.op_type == "ReduceMean" and n.name.endswith("/ReduceMean")]:
        P = rm.name[:-len("ReduceMean")]
        parts = {k: by_name.get(P + k) for k in ("Sub", "Mul_2", "ReduceMean_2", "Add_2", "Sqrt", "Div_1")}
        if any(v is None for v in parts.values()):
            continue
        x = rm.input[0]
        eps = [i for i in parts["Add_2"].input if i != parts["ReduceMean_2"].output[0]][0]
        # One normalisation can feed several affine heads: the exporter merged the identical
        # mean/var of the parallel resblocks that normalise the same tensor.
        heads = []
        for mulj in consumers[parts["Div_1"].output[0]]:
            assert mulj.op_type == "Mul", mulj.name
            addj = consumers[mulj.output[0]]
            assert len(addj) == 1 and addj[0].op_type == "Add", mulj.name
            heads.append((mulj, addj[0]))
        new = []
        T = lambda b: nm.tmp(b)
        xw, mu, x2w, m2, mu2, var, ve, sd = (T(b) for b in ("xw", "mu", "x2w", "m2", "mu2", "var", "ve", "sd"))
        if mask_norm:
            new.append(helper.make_node("Mul", [x, w_of(x)], [xw]))
        else:
            xw = x
        new.append(helper.make_node("GlobalAveragePool", [xw], [mu]))
        if var2:        # centred two-pass variance (fp16 safe): var = GAP(w (x-mu)^2)
            xc, xcw = T("xc"), T("xcw")
            new += [helper.make_node("Sub", [x, mu], [xc]),
                    helper.make_node("Mul", [xc, w_of(x)], [xcw]) if mask_norm else helper.make_node("Identity", [xc], [xcw]),
                    helper.make_node("Mul", [xcw, xc], [x2w]),
                    helper.make_node("GlobalAveragePool", [x2w], [var])]
        else:           # one pass: var = E_w[x^2] - mu^2 (13% faster, but worse in fp16)
            new += [helper.make_node("Mul", [xw, x], [x2w]),
                    helper.make_node("GlobalAveragePool", [x2w], [m2]),
                    helper.make_node("Mul", [mu, mu], [mu2]),
                    helper.make_node("Sub", [m2, mu2], [var])]
        new += [helper.make_node("Add", [var, eps], [ve]),
                helper.make_node("Sqrt", [ve], [sd])]
        for k in parts.values():
            remove.add(id(k))
        remove.add(id(rm))
        for mul3, add3 in heads:
            g1 = [i for i in mul3.input if i != parts["Div_1"].output[0]][0]
            beta = [i for i in add3.input if i != mul3.output[0]][0]
            a_out = add3.output[0]
            remove.add(id(mul3)); remove.add(id(add3))
            A, muA, B = T("A"), T("muA"), T("B")
            new += [helper.make_node("Div", [g1, sd], [A]),
                    helper.make_node("Mul", [mu, A], [muA]),
                    helper.make_node("Sub", [beta, muA], [B])]
            cons = consumers.get(a_out, [])
            snk = [c for c in cons if c.op_type == "Mul" and any(i in init and i != a_out for i in c.input)]
            fused = False
            if sin_mode in ("rr", "rr_npu") and len(snk) == 1 and len(cons) == 2:
                ma = snk[0]
                alpha = numpy_helper.to_array(init[[i for i in ma.input if i != a_out][0]]).astype(np.float64)
                sn = consumers[ma.output[0]][0]
                pw = consumers[sn.output[0]][0] if sn.op_type == "Sin" else None
                mr = consumers[pw.output[0]][0] if pw is not None and pw.op_type == "Pow" else None
                ad = [c for c in cons if c.op_type == "Add"]
                if mr is not None and mr.op_type == "Mul" and ad and consumers[mr.output[0]][0] is ad[0]:
                    ad = ad[0]
                    cv = consumers[ad.output[0]]
                    if len(cv) == 1 and cv[0].op_type == "Conv":
                        conv = cv[0]
                        k = nm.const((alpha / np.pi).astype(np.float32))
                        A2, B2, xa, q, rq, d, u = (T(b) for b in ("A2", "B2", "xa", "q", "rq", "d", "u"))
                        new += [helper.make_node("Mul", [A, k], [A2]), helper.make_node("Mul", [B, k], [B2]),
                                helper.make_node("Mul", [x, A2], [xa]), helper.make_node("Add", [xa, B2], [q])]
                        if sin_mode == "rr":
                            new.append(helper.make_node("Round", [q], [rq]))
                        else:
                            qi = T("qi")
                            new += [helper.make_node("Cast", [q], [qi], to=TensorProto.INT16),
                                    helper.make_node("Cast", [qi], [rq], to=TensorProto.FLOAT)]
                        new += [helper.make_node("Sub", [q, rq], [d]), helper.make_node("Mul", [d, d], [u])]
                        c, _ = sin2_over_pi_coeffs(deg)
                        acc = None
                        for ci in c[::-1]:           # Horner: ((c3 u + c2) u + c1) u
                            t1 = T("hm")
                            if acc is None:
                                new.append(helper.make_node("Mul", [u, nm.const(np.float32(ci))], [t1]))
                            else:
                                t0 = T("ha")
                                new.append(helper.make_node("Add", [acc, nm.const(np.float32(ci))], [t0]))
                                new.append(helper.make_node("Mul", [t0, u], [t1]))
                            acc = t1
                        new.append(helper.make_node("Add", [q, acc], [ad.output[0]]))
                        W = numpy_helper.to_array(init[conv.input[1]]).astype(np.float64)
                        conv.input[1] = nm.const((W * (np.pi / alpha.reshape(1, -1, 1))).astype(np.float32))
                        for kk in (ma, sn, pw, mr, ad):
                            remove.add(id(kk))
                        fused = True
                        n_sn += 1
            if not fused:
                xa = T("xa")
                new += [helper.make_node("Mul", [x, A], [xa]), helper.make_node("Add", [xa, B], [a_out])]
            n_ad += 1
        insert_before[id(rm)] = new
    out = []
    for n in nodes:
        if id(n) in insert_before:
            out.extend(insert_before[id(n)])
        if id(n) not in remove:
            out.append(n)
    del g.node[:]
    g.node.extend(out)
    return n_ad, n_sn


def build(src, T, sin_mode="rr_npu", mask_norm=True, mask_conv=True, sin_deg=3, split=True, fuse=True,
          mask_conv_hi=True, var2=True):
    m = onnx.load(str(src))
    m = Extractor(onnx.shape_inference.infer_shapes(m)).extract_model(["asr", "F0", "N", "s", "har"], [X_POST])
    g = m.graph
    L = {"T": T, "2T": 2 * T, "20T": 20 * T, "120T1": 120 * T + 1}
    by_len = {v: k for k, v in L.items()}
    shapes = {"asr": [1, 512, T], "F0": [1, 2 * T], "N": [1, 2 * T], "s": [1, 128], "har": [1, 22, 120 * T + 1]}
    del g.input[:]
    for n, sh in shapes.items():
        g.input.append(helper.make_tensor_value_info(n, TensorProto.FLOAT, sh))
    for k, n in L.items():
        if mask_conv:
            g.input.append(helper.make_tensor_value_info(f"m_{k}", TensorProto.FLOAT, [1, 1, n]))
        if mask_norm:
            g.input.append(helper.make_tensor_value_info(f"w_{k}", TensorProto.FLOAT, [1, 1, n]))
    del g.output[:]
    g.output.append(helper.make_tensor_value_info("x_post", TensorProto.FLOAT, [1, 22, 120 * T + 1]))
    for n in g.node:
        for i, o in enumerate(n.output):
            if o == X_POST:
                n.output[i] = "x_post"
    del g.value_info[:]
    mi = onnx.shape_inference.infer_shapes(m)
    shp = {v.name: [d.dim_value for d in v.type.tensor_type.shape.dim]
           for v in list(mi.graph.value_info) + list(mi.graph.input)}

    def tlen(x):
        s = shp.get(x)
        assert s and len(s) == 3 and s[2] in by_len, (x, s)
        return by_len[s[2]]

    nm = Names(g)
    if fuse:
        n_ad, n_sn = fuse_norm_snake(g, nm, mask_norm, sin_mode, sin_deg, lambda x: f"w_{tlen(x)}", var2)
        print(f"fused {n_ad} norms, {n_sn} snakes")
    nodes = list(g.node)
    consumers = {}
    for n in nodes:
        for i in n.input:
            consumers.setdefault(i, []).append(n)
    new, masked, skip = [], {}, set()
    rp_out = None
    cnt = dict(rm=0, mask=0, sin=0, deconv=0)
    for n in nodes:
        if id(n) in skip:
            continue
        if n.op_type == "ReduceMean":
            x = n.input[0]
            if mask_norm:
                xw = nm.tmp("xw")
                new.append(helper.make_node("Mul", [x, f"w_{tlen(x)}"], [xw], name=n.name + "/npu2_w"))
                x = xw
            new.append(helper.make_node("GlobalAveragePool", [x], [n.output[0]], name=n.name + "/npu2_gap"))
            cnt["rm"] += 1
            continue
        if n.op_type in ("Conv", "ConvTranspose") and mask_conv:
            ks = [helper.get_attribute_value(a) for a in n.attribute if a.name == "kernel_shape"][0]
            x = n.input[0]
            if ks[0] > 1 and x not in shapes and (mask_conv_hi or tlen(x) != "120T1"):
                if x not in masked:
                    xm = nm.tmp("masked")
                    new.append(helper.make_node("Mul", [x, f"m_{tlen(x)}"], [xm], name=n.name + "/npu2_mask"))
                    masked[x] = xm
                    cnt["mask"] += 1
                n.input[0] = masked[x]
        if n.op_type == "ConvTranspose":
            at = {a.name: helper.get_attribute_value(a) for a in n.attribute}
            S = at["strides"][0]
            if S == 6 and split:
                # ups.1 as POLYPHASE: one Conv with 6 x Cout channels on the 20T grid; the CPU
                # interleaves the phases (an NPU reshape of that size falls back / overflows).
                K, P = at["kernel_shape"][0], at["pads"][0]
                assert (K, P) == (12, 3), (K, P)
                w = numpy_helper.to_array(next(i for i in g.initializer if i.name == n.input[1]))  # (Cin, Cout, K)
                cin, cout, _ = w.shape
                wp = np.zeros((6 * cout, cin, 3), np.float32)
                for ph in range(6):
                    blk = wp[ph * cout:(ph + 1) * cout]
                    if ph < 3:      # y[6q+ph] = x[q] w[ph+3] + x[q-1] w[ph+9]
                        blk[:, :, 1] = w[:, :, ph + 3].T
                        blk[:, :, 0] = w[:, :, ph + 9].T
                    else:           # y[6q+ph] = x[q] w[ph+3] + x[q+1] w[ph-3]
                        blk[:, :, 1] = w[:, :, ph + 3].T
                        blk[:, :, 2] = w[:, :, ph - 3].T
                bias = numpy_helper.to_array(next(i for i in g.initializer if i.name == n.input[2]))
                new.append(helper.make_node("Conv", [n.input[0], nm.const(wp), nm.const(np.tile(bias, 6))], ["ups1_poly"],
                                            name=n.name + "/npu2_poly", kernel_shape=[3], pads=[1, 1]))
                cnt["deconv"] += 1
                continue
            if S not in (1, 2, 4, 8):          # ups.0, stride 10 (adapted from kokoro-server's build.py)
                K, P = at["kernel_shape"][0], at["pads"][0]
                assert at["pads"][0] == at["pads"][1] and at.get("group", 1) == 1
                assert list(at.get("output_padding", [0])) == [0]
                w = numpy_helper.to_array(next(i for i in g.initializer if i.name == n.input[1]))
                wn = nm.const(np.flip(w.transpose(1, 0, 2), 2).copy())
                r4, pd, fl = nm.tmp("r4"), nm.tmp("pad"), nm.tmp("flat")
                new.append(helper.make_node("Reshape", [n.input[0], nm.const([0, 0, -1, 1], np.int64)], [r4]))
                new.append(helper.make_node("Pad", [r4, nm.const([0, 0, 0, 0, 0, 0, 0, S - 1], np.int64)], [pd]))
                new.append(helper.make_node("Reshape", [pd, nm.const([0, 0, -1], np.int64)], [fl]))
                new.append(helper.make_node("Conv", [fl, wn] + list(n.input[2:]), list(n.output),
                                            name=n.name + "/npu2_conv", kernel_shape=[K],
                                            pads=[K - 1 - P, K - S - P], strides=[1], dilations=[1]))
                cnt["deconv"] += 1
                continue
        if split and n.op_type == "Pad" and "reflection_pad" in n.name:
            rp_out = n.output[0]
            continue
        if n.op_type == "Sin" and ("resblocks" in n.name or "noise_res" in n.name):
            cons = consumers[n.output[0]]
            assert len(cons) == 1 and cons[0].op_type == "Pow", n.name
            skip.add(id(cons[0]))
            new.extend(sin2_nodes(nm, n.input[0], cons[0].output[0], sin_mode, sin_deg))
            cnt["sin"] += 1
            continue
        new.append(n)
    del g.node[:]
    g.node.extend(new)
    for op in m.opset_import:          # librknnc accepts opset <= 19; nothing here changed after 17
        if op.domain in ("", "ai.onnx"):
            op.version = 17
    if not split:
        m = onnx.shape_inference.infer_shapes(m)
        onnx.checker.check_model(m)
        print(f"T={T} sin={sin_mode} mask_norm={mask_norm} mask_conv={mask_conv}: {cnt}; {len(m.graph.node)} nodes")
        return m
    assert rp_out is not None
    L = 120 * T + 1
    # A: everything at the frame rates, ending in the polyphase ups.1 output
    g.output.append(helper.make_tensor_value_info("ups1_poly", TensorProto.FLOAT, [1, 768, 20 * T]))
    g.input.append(helper.make_tensor_value_info(rp_out, TensorProto.FLOAT, [1, 128, L]))
    a_in = [i.name for i in g.input if not i.name.endswith("120T1") and i.name != rp_out]
    A = Extractor(m).extract_model(a_in, ["ups1_poly"])
    used = {x for n in A.graph.node for x in n.input}
    keep = [v for v in A.graph.input if v.name in used]
    del A.graph.input[:]; A.graph.input.extend(keep)
    # B: the 120T+1 stage. x_after (after the reflection pad) and the masks come in as 4-D views.
    b_in = [rp_out, "s", "har"] + [i.name for i in g.input if i.name.endswith("120T1")]
    B = Extractor(m).extract_model(b_in, ["x_post"])
    Hb, Wb, Lb = b_shape(T)
    for v in list(B.graph.input) + list(B.graph.output):
        dims = v.type.tensor_type.shape.dim
        if len(dims) == 3 and dims[2].dim_value == L:
            dims[2].dim_value = Lb
    del B.graph.value_info[:]
    for n in B.graph.node:
        for i, x in enumerate(n.input):
            if x == rp_out:
                n.input[i] = "x_after"
    for v in B.graph.input:
        if v.name == rp_out:
            v.name = "x_after"
    out = []
    for name, mm in (("A", A), ("B", B)):
        mm = onnx.shape_inference.infer_shapes(mm)
        nv = view4d(mm, {"x_after", "m_120T1", "w_120T1"}) if name == "B" and Lb > MAXW else 0
        mm = onnx.shape_inference.infer_shapes(mm)
        onnx.checker.check_model(mm)
        print(f"{name}: T={T} L={L} B view={Hb}x{Wb}={Lb} sin={sin_mode} mask_norm={mask_norm} mask_conv={mask_conv}: "
              f"{cnt}, {nv} 4D views; {len(mm.graph.node)} nodes, inputs {[i.name for i in mm.graph.input]}")
        out.append(mm)
    return out


MAXW = 8191                 # 13-bit width field in the NPU's broadcast/elementwise registers
ELEMENTWISE = {"Add", "Sub", "Mul", "Div", "Sqrt", "Abs", "Cast", "Round", "LeakyRelu", "Clip", "Exp", "Sin",
               "Reciprocal", "Pow", "Neg", "Relu", "Sigmoid", "Tanh"}


def b_shape(T):
    """The 120T+1 stage runs on a padded length L' = H * W (zeros past 120T+1, masked) with W a
    multiple of 32: elementwise ops on such views measured ~35% faster than on odd widths."""
    L = 120 * T + 1
    H = -(-L // 8160)
    W = -(-L // (H * 32)) * 32
    return H, W, H * W


def factor(L):
    """L = H * W with W <= MAXW, smallest H."""
    for H in range(1, 64):
        if L % H == 0 and L // H <= MAXW and (L // H) % 32 == 0:
            return H, L // H
    for H in range(2, L):
        if L % H == 0 and L // H <= MAXW:
            return H, L // H
    raise SystemExit(f"length {L} has no factorisation with W <= {MAXW}: pick another bucket")


def view4d(m, inputs4d=()):
    """librknnc encodes a broadcast elementwise op's width in a 13-bit field: past
    8191 frames of time axis (here 120T+1 for T > 68) the compiled model hangs
    the NPU. Convolutions and GlobalAveragePool are fine. So every elementwise op
    on a (1,C,L) tensor with L > 8191 runs on a (1,C,H,W) view of it (H*W = L,
    the same memory: a free Reshape), its per-channel operands as (1,C,1,1)."""
    g = m.graph
    shp = {v.name: [d.dim_value for d in v.type.tensor_type.shape.dim]
           for v in list(g.value_info) + list(g.input) + list(g.output)}
    for i in g.initializer:
        shp[i.name] = list(i.dims)
    for v in g.input:                   # inputs given as 4-D views: their 3-D form never exists
        if v.name in inputs4d:
            s3 = [d.dim_value for d in v.type.tensor_type.shape.dim]
            v.CopyFrom(helper.make_tensor_value_info(v.name, TensorProto.FLOAT, [1, s3[1], *factor(s3[2])]))
    big = lambda x: len(shp.get(x, [])) == 3 and shp[x][2] > MAXW
    small3 = lambda x: len(shp.get(x, [])) == 3 and shp[x][2] == 1
    rep4, rep3, new = {}, {}, []
    for v in g.input:
        if len(v.type.tensor_type.shape.dim) == 4:
            rep4[v.name] = v.name
            rep3[v.name] = None if False else v.name     # 3-D form made by Reshape on demand
    four_d_inputs = {v.name for v in g.input if len(v.type.tensor_type.shape.dim) == 4}
    k = [0]

    def const(v):
        k[0] += 1
        name = f"npu2_v{k[0]}"
        g.initializer.append(numpy_helper.from_array(np.asarray(v, np.int64), name))
        return name

    def as4(x):
        if x in rep4:
            return rep4[x]
        s = shp[x]
        tgt = [1, s[1], *factor(s[2])] if s[2] > 1 else [1, s[1], 1, 1]
        y = x + "__4d"
        new.append(helper.make_node("Reshape", [x, const(tgt)], [y]))
        rep4[x] = y
        return y

    def as3(x):
        """The 3-D tensor x, materialised from its 4-D form when only that exists."""
        if x in rep3 and rep3[x] is not None:
            y = x + "__3d" if x in four_d_inputs else x
            new.append(helper.make_node("Reshape", [rep3[x], const(shp[x])], [y]))
            rep3[x] = None
            if y != x:
                rename3[x] = y
        return rename3.get(x, x)
    rename3 = {}

    n_view = 0
    for n in list(g.node):
        ins = list(n.input)
        if n.op_type in ELEMENTWISE and any(big(x) or x in rep3 for x in ins if x):
            n_view += 1
            for i, x in enumerate(ins):
                if not x:
                    continue
                if x in rep3:
                    n.input[i] = rep4[x]
                elif big(x) or small3(x):
                    n.input[i] = as4(x)
                else:
                    assert len(shp.get(x, [])) < 3, (n.name, x, shp.get(x))
            for j, y in enumerate(n.output):
                y4 = y + "__4d"
                n.output[j] = y4
                rep4[y], rep3[y] = y4, y4
            new.append(n)
            continue
        if n.op_type == "GlobalAveragePool" and ins[0] in rep3:
            n.input[0] = rep4[ins[0]]
            y = n.output[0]
            n.output[0] = y + "__4d"
            rep4[y], rep3[y] = y + "__4d", y + "__4d"
            new.append(n)
            continue
        for i, x in enumerate(ins):
            if x in rep3:
                n.input[i] = as3(x)
        new.append(n)
    for o in g.output:
        if o.name in rep3:
            as3(o.name)
    del g.node[:]
    g.node.extend(new)
    return n_view


if __name__ == "__main__":
    from pathlib import Path
    ap = argparse.ArgumentParser(description="NPU decoder graphs (A and B) for one bucket length")
    ap.add_argument("--models", default="models", help="directory with dec.onnx (build/extract.py)")
    ap.add_argument("--T", type=int, required=True, help="bucket length in frames (25 ms each)")
    ap.add_argument("--out", required=True, help="writes <out>_A.onnx and <out>_B.onnx")
    ap.add_argument("--sin", default="rr_npu", choices=["rr_npu", "rr", "clip", "cpu"],
                    help="rr_npu: range-reduced, NPU rounding (deployed); rr: same with Round (fp32 checks); "
                         "clip: the earlier clipped polynomial; cpu: exact Sin (runs on the CPU)")
    ap.add_argument("--sin-deg", type=int, default=3, help="polynomial degree in r^2")
    ap.add_argument("--no-fuse", action="store_true", help="keep each AdaIN and Snake as separate ops")
    ap.add_argument("--var1", action="store_true", help="one-pass variance (faster, worse in fp16)")
    ap.add_argument("--no-mask-norm", action="store_true", help="(experiments) plain means over the padded length")
    ap.add_argument("--no-mask-conv", action="store_true", help="(experiments) no masks on conv inputs")
    ap.add_argument("--no-mask-conv-hi", action="store_true", help="no masks on the 120T+1-rate conv inputs (~5%% faster, less exact)")
    ap.add_argument("--coeffs", action="store_true", help="print the sin^2 fit error per degree and exit")
    a = ap.parse_args()
    if a.coeffs:
        for d in range(3, 9):
            print(d, "max err %.3g" % sin2_coeffs(d)[1])
        raise SystemExit
    A, B = build(Path(a.models) / "dec.onnx", a.T, a.sin, not a.no_mask_norm, not a.no_mask_conv, a.sin_deg,
                 fuse=not a.no_fuse, mask_conv_hi=not a.no_mask_conv_hi, var2=not a.var1)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    onnx.save(A, a.out.replace(".onnx", "_A.onnx"))
    onnx.save(B, a.out.replace(".onnx", "_B.onnx"))
