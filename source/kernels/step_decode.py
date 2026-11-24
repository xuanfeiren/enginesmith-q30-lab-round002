"""Whole Qwen3-0.6B decode step in one NEFF (LNC=1): token -> embedding -> 28 layers -> final norm -> LM head ->
argmax (ties to the lowest id) -> next token, with position and token history kept on the device.

Inputs (beyond those of layer_decode._layers):
  pos    [1, 1] int32     position of the input token (read on the device)
  plen   [1, 1] int32     prompt length
  prompt [TMAX, 1] int32  prompt tokens (host writes once per request)
  toks   [TMAX+1, 1] int32 token history: the step writes its argmax to toks[pos + 1] (in place)
  embed  [V, 1024] bf16   token embedding
  normf  [128, 8] fp32    final RMSNorm weight, column layout
  wlm    [128, NV*8*128]  ws_layout(lm_head)   (V = 151,936 = 1187 * 128)
Input token = prompt[pos] if pos < plen else toks[pos]: the whole request (prompt fed token by token, then
generation) is a chain of identical executions; the host writes the prompt once and reads toks once.
Outputs: output_0 = pos + 1 [1, 1] int32 (ping-pong with `pos` in the host's two tensor sets),
         output_1 = logits column layout [128, NV] fp32 (only in the *_dbg variant).
"""
import nki
import nki.language as nl
import nki.isa as nisa
from layer_decode import _layer, _consts, _gemv, _rms_col, HT, HID

BIGC = 1048576.0   # 2^20 > vocab: score = BIGC - id, max score <=> min id


def _token_in(pos, plen, prompt, toks, p_sb):
    """token id (int32 [1,1] SBUF) = prompt[pos] if pos < plen else toks[pos]."""
    preg = nisa.register_alloc()
    nisa.register_load(preg, p_sb)
    pt = nl.ndarray((1, 2), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=pt[:, 0:1], src=prompt.ap(pattern=[[1, 1], [1, 1]], offset=0, scalar_offset=preg, indirect_dim=0))
    nisa.dma_copy(dst=pt[:, 1:2], src=toks.ap(pattern=[[1, 1], [1, 1]], offset=0, scalar_offset=preg, indirect_dim=0))
    pl = nl.ndarray((1, 2), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=pl[:, 0:1], src=plen)
    nisa.tensor_copy(dst=pl[:, 1:2], src=p_sb)
    f = nl.ndarray((1, 2), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=f, src=pt)
    g = nl.ndarray((1, 2), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=g, src=pl)
    sel = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)        # 1.0 if pos < plen
    nisa.tensor_tensor(dst=sel, data1=g[:, 1:2], data2=g[:, 0:1], op=nl.less)
    d = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=d, data1=f[:, 0:1], data2=f[:, 1:2], op=nl.subtract)
    nisa.tensor_tensor(dst=d, data1=d, data2=sel, op=nl.multiply)
    tf = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=tf, data1=d, data2=f[:, 1:2], op=nl.add)   # gen + sel * (prompt - gen)
    ti = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=ti, src=tf)
    return ti


def _embed(embed, t_sb):
    treg = nisa.register_alloc()
    nisa.register_load(treg, t_sb)
    eb = nl.ndarray((128, HT), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=eb, src=embed.ap(pattern=[[1, 128], [128, HT]], offset=0, scalar_offset=treg, indirect_dim=0))
    h = nl.ndarray((128, HT), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=h, src=eb)
    return h


def _lm_argmax(xf, wlm, NV, SL, hw, ident_f):
    """logits col [128, NV] fp32 (SBUF) and the argmax token as fp32 [1,1] (lowest id among equal maxima)."""
    lg = nl.ndarray((128, NV), dtype=nl.float32, buffer=nl.sbuf)
    qi = 0
    c0 = 0
    while c0 < NV:
        n = 512
        if NV - c0 < 512:
            n = NV - c0
        ps = nl.ndarray((128, 512), dtype=nl.float32, buffer=nl.psum)
        qi = _gemv(xf, wlm, HT, c0, c0 + n, ps, 0, SL, hw, qi)
        nisa.tensor_copy(dst=lg[:, c0:c0 + n], src=ps[:, 0:n])
        c0 += n
    # per partition: max and the lowest id reaching it (id = col*128 + p)
    pm = nl.ndarray((128, 2), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=pm[:, 0:1], op=nl.maximum, data=lg, axis=1)
    sc = nl.ndarray((128, NV), dtype=nl.float32, buffer=nl.sbuf)
    nisa.iota(dst=sc, pattern=[[128, NV]], offset=0, channel_multiplier=1)
    nisa.tensor_scalar(dst=sc, data=sc, op0=nl.multiply, operand0=-1.0, op1=nl.add, operand1=BIGC)
    eq = nl.ndarray((128, NV), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=eq, data=lg, op0=nl.equal, operand0=pm[:, 0:1])
    nisa.tensor_tensor(dst=eq, data1=eq, data2=sc, op=nl.multiply)
    nisa.tensor_reduce(dst=pm[:, 1:2], op=nl.maximum, data=eq, axis=1)          # BIGC - lowest id
    # across partitions: transpose both columns to rows on partition 0
    rows = nl.ndarray((1, 2, 128), dtype=nl.float32, buffer=nl.sbuf)
    for c in range(2):
        tp = nl.ndarray((1, 128), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=tp, data=pm[:, c:c + 1])
        nisa.tensor_copy(dst=rows[:, c, :], src=tp)
    gm = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=gm, op=nl.maximum, data=rows[:, 0, :], axis=1)
    e2 = nl.ndarray((1, 128), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=e2, data=rows[:, 0, :], op0=nl.equal, operand0=gm)
    nisa.tensor_tensor(dst=e2, data1=e2, data2=rows[:, 1, :], op=nl.multiply)
    bs = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=bs, op=nl.maximum, data=e2, axis=1)
    tokf = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=tokf, data=bs, op0=nl.multiply, operand0=-1.0, op1=nl.add, operand1=BIGC)
    return lg, tokf


def _step(pos, plen, prompt, toks, embed, ln1, ln2, qn, kn, wqkv, wo, wgu, wd, normf, wlm, rot, cos, sin,
          kcache, vcache, TB, SL, hw, dbg):
    L = wqkv.shape[0]
    NV = embed.shape[0] // 128
    # position: register, broadcast float column, RoPE columns
    p_sb = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=p_sb, src=pos)
    preg = nisa.register_alloc()
    nisa.register_load(preg, p_sb)
    pb_i = nl.ndarray((128, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=pb_i, src=pos.ap(pattern=[[0, 128], [1, 1]], offset=0))
    pb = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=pb, src=pb_i)
    cs_col = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
    sn_col = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
    for half in range(2):
        nisa.dma_copy(dst=cs_col[half * 64:(half + 1) * 64, :],
                      src=cos.ap(pattern=[[1, 64], [1, 1]], offset=0, scalar_offset=preg, indirect_dim=0))
        nisa.dma_copy(dst=sn_col[half * 64:(half + 1) * 64, :],
                      src=sin.ap(pattern=[[1, 64], [1, 1]], offset=0, scalar_offset=preg, indirect_dim=0))
    ones, ident, idf, ones16 = _consts()
    rot_sb = nl.ndarray((128, 128), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=rot_sb, src=rot)

    t_sb = _token_in(pos, plen, prompt, toks, p_sb)
    h = _embed(embed, t_sb)
    for l in range(L):
        h = _layer(h, l, ln1, ln2, qn, kn, wqkv, wo, wgu, wd, rot_sb, cos, sin, kcache, vcache,
                   preg, pb, cs_col, sn_col, ones, ident, idf, ones16, TB, SL, hw)
    wf = nl.ndarray((128, HT), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=wf, src=normf)
    xf = _rms_col(h, wf, HT, HID, ones, nl.bfloat16)
    lg, tokf = _lm_argmax(xf, wlm, NV, SL, hw, idf)

    # outputs: toks[pos + 1] = argmax, pos + 1
    tok_i = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=tok_i, src=tokf)
    nisa.dma_copy(dst=toks.ap(pattern=[[1, 1], [1, 1]], offset=1, scalar_offset=preg, indirect_dim=0), src=tok_i)
    p1 = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=p1, data=p_sb, op0=nl.add, operand0=1)
    pos_out = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=pos_out, src=p1)
    if dbg:
        lg_out = nl.ndarray((128, NV), dtype=nl.float32, buffer=nl.shared_hbm)
        nisa.dma_copy(dst=lg_out, src=lg)
        return pos_out, lg_out
    return pos_out


@nki.jit
def step_b512(pos, plen, prompt, toks, embed, ln1, ln2, qn, kn, wqkv, wo, wgu, wd, normf, wlm, rot, cos, sin,
              kcache, vcache):
    return _step(pos, plen, prompt, toks, embed, ln1, ln2, qn, kn, wqkv, wo, wgu, wd, normf, wlm, rot, cos, sin,
                 kcache, vcache, 512, 4, 0, 0)


@nki.jit
def step_b512_dbg(pos, plen, prompt, toks, embed, ln1, ln2, qn, kn, wqkv, wo, wgu, wd, normf, wlm, rot, cos, sin,
                  kcache, vcache):
    return _step(pos, plen, prompt, toks, embed, ln1, ln2, qn, kn, wqkv, wo, wgu, wd, normf, wlm, rot, cos, sin,
                 kcache, vcache, 512, 4, 0, 1)
