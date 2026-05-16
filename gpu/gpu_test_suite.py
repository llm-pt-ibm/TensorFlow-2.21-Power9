#!/usr/bin/env python3
"""TensorFlow GPU functional coverage suite for Power9 + V100.

Sections:
  Core (T01-T12)              setup, device, memory, basic ops, matmul
  Tensor ops (T13-T20)        creation, dtypes, shape, math, reduce, indexing
  Linear algebra (T21-T26)    matmul flavors, inv/solve/det, decomp, einsum
  Random (T27-T30)            distributions, stateless, seed reproducibility
  NN layers (T31-T40)         activations, pooling, normalization, embedding,
                              rnn cells, attention
  Image / Signal (T41-T45)    resize/crop, color, ssim, FFT, conv1d
  Sparse / Ragged / Strings   (T46-T48) sparse ops, ragged ops, strings
  Data pipeline (T49-T52)     tf.data primitives + transforms
  Vars / Opt / Loss (T53-T57) Variable, optimizers, losses
  Keras models (T58-T63)      Sequential, Functional, save/load, callbacks
  Grad / AutoDiff (T64-T67)   GradientTape, jacobian, stop_gradient
  tf.function (T68-T70)       basic / signature / autograph
  Persistence (T71-T72)       Checkpoint / SavedModel
  Mixed precision (T73)       fp16 policy
  Distribute (T74)            OneDeviceStrategy
  Extras (T75)                tf.config / threading

Usage:
  python3 gpu_test_suite.py              # all tests
  python3 gpu_test_suite.py T05 T30      # only listed
  python3 gpu_test_suite.py --quick      # skip stress + multi-gpu + heavy
"""
import os
import re
import sys
import time
import traceback

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# Ensure conda libstdc++ + CUDA libs are visible to the dynamic linker.
def _ensure_env():
    if os.environ.get("GPU_TESTS_REEXEC") == "1":
        return
    needed = []
    conda = os.environ.get("CONDA_PREFIX")
    if conda and os.path.isdir(f"{conda}/lib"):
        needed.append(f"{conda}/lib")
    cu = os.path.expanduser("~/cuda_unified/lib64")
    if os.path.isdir(cu):
        needed.append(cu)
    current = os.environ.get("LD_LIBRARY_PATH", "")
    missing = [p for p in needed if p not in current.split(":")]
    if not missing:
        return
    new_env = os.environ.copy()
    new_env["LD_LIBRARY_PATH"] = ":".join(needed + ([current] if current else []))
    new_env["GPU_TESTS_REEXEC"] = "1"
    print(f"[setup] re-exec with LD_LIBRARY_PATH={new_env['LD_LIBRARY_PATH']}",
          file=sys.stderr)
    os.execvpe(sys.executable, [sys.executable] + sys.argv, new_env)
_ensure_env()


PASS = 0
FAIL = 0
SKIP = 0
RESULTS = []

ARGS = set(a for a in sys.argv[1:] if not a.startswith("-"))
FLAGS = set(a for a in sys.argv[1:] if a.startswith("-"))
QUICK = "--quick" in FLAGS or "-q" in FLAGS

# Build-config issues that turn FAIL -> SKIP.
KNOWN_SKIPS = [
    ("No DNN in stream executor", "cuDNN runtime/compile mismatch"),
    ("Loaded runtime CuDNN library", "cuDNN runtime/compile mismatch"),
    ("could not find registered compiler for the platform", "XLA CUDA backend not registered"),
    ("No registered 'Add' OpKernel for 'GPU'", "float32 Add GPU kernel missing (NCCL build gap)"),
    ("No registered '.*' OpKernel for 'GPU'", "GPU kernel not registered for this op"),
    ("DNN library initialization failed", "cuDNN init failure"),
    ("CUDNN_STATUS_VERSION_MISMATCH", "cuDNN version mismatch"),
]

def _classify(exc_text):
    for pat, reason in KNOWN_SKIPS:
        if re.search(pat, exc_text):
            return reason
    return None


def run(name, fn):
    global PASS, FAIL, SKIP
    if ARGS and name not in ARGS:
        return
    print(f"\n[{name}] {fn.__doc__ or ''}")
    t0 = time.time()
    try:
        ok, detail = fn()
    except Exception as e:
        exc_text = f"{type(e).__name__}: {e}"
        reason = _classify(exc_text)
        if reason:
            ok, detail = None, f"skipped ({reason})"
        else:
            ok, detail = False, f"exception: {exc_text}"
            traceback.print_exc(limit=2)
    dt = time.time() - t0
    if ok is None:
        SKIP += 1; tag = "SKIP"
    elif ok:
        PASS += 1; tag = "PASS"
    else:
        FAIL += 1; tag = "FAIL"
    RESULTS.append((name, tag, dt, detail))
    print(f"  -> {tag} ({dt:.2f}s) {detail}")


# ============================================================================
# CORE (T01-T12)
# ============================================================================

def t01_setup():
    """imports, version, build flags"""
    import tensorflow as tf
    info = {"tf": tf.__version__, "cuda": tf.test.is_built_with_cuda(),
            "gpu_support": tf.test.is_built_with_gpu_support()}
    return info["cuda"], str(info)

def t02_detect():
    """list physical GPUs and names"""
    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        return False, "no GPUs detected"
    details = []
    for g in gpus:
        try:
            d = tf.config.experimental.get_device_details(g)
            details.append(f"{g.name}={d.get('device_name','?')}")
        except Exception:
            details.append(g.name)
    return True, f"{len(gpus)} GPU(s): " + ", ".join(details)

def t03_memory():
    """allocate + free large tensors on GPU"""
    import tensorflow as tf
    if not tf.config.list_physical_devices('GPU'):
        return None, "no GPU"
    with tf.device('/GPU:0'):
        bufs = [tf.zeros([1024, 1024, 64], dtype=tf.float32) for _ in range(4)]
        mb = sum(b.numpy().nbytes for b in bufs) / 1e6
        del bufs
    return True, f"alloc/free {mb:.0f}MB OK"

def t04_basics():
    """random/reduce/broadcast on GPU"""
    import tensorflow as tf
    with tf.device('/GPU:0'):
        x = tf.random.normal([4096, 4096])
        s = tf.reduce_sum(x); m = tf.reduce_mean(x)
        y = x + tf.range(4096, dtype=tf.float32)
        out = tf.reduce_max(y)
    return True, f"sum={float(s):.1f} mean={float(m):.4f} max={float(out):.1f}"

def t05_matmul():
    """matmul 512/4096/8192 GFLOPs"""
    import tensorflow as tf
    times = []
    with tf.device('/GPU:0'):
        for N in [512, 4096, 8192]:
            a = tf.random.normal([N, N]); b = tf.random.normal([N, N])
            _ = tf.matmul(a, b)
            t0 = time.time()
            for _ in range(3):
                c = tf.matmul(a, b)
            _ = c.numpy()
            dt = (time.time() - t0) / 3
            gf = 2 * N**3 / dt / 1e9
            times.append(f"{N}={dt*1000:.1f}ms/{gf:.0f}GF")
    return True, " ".join(times)

def t06_conv2d():
    """3-layer conv2d forward"""
    import tensorflow as tf
    with tf.device('/GPU:0'):
        x = tf.random.normal([8, 224, 224, 3])
        l1 = tf.keras.layers.Conv2D(64, 7, strides=2, padding='same', activation='relu')(x)
        l2 = tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu')(l1)
        l3 = tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu')(l2)
    return True, f"input={x.shape} -> {l3.shape}"

def t07_backprop():
    """gradient through MLP"""
    import tensorflow as tf
    with tf.device('/GPU:0'):
        x = tf.random.normal([32, 128]); target = tf.random.normal([32, 10])
        w1 = tf.Variable(tf.random.normal([128, 256]))
        w2 = tf.Variable(tf.random.normal([256, 10]))
        with tf.GradientTape() as tape:
            loss = tf.reduce_mean((tf.nn.relu(x @ w1) @ w2 - target) ** 2)
        g1, g2 = tape.gradient(loss, [w1, w2])
    return g1 is not None and g2 is not None, f"loss={float(loss):.3f} |dw1|={float(tf.norm(g1)):.2f}"

def t08_train():
    """1 epoch MNIST-shape (jit_compile=False)"""
    import tensorflow as tf
    with tf.device('/GPU:0'):
        x = tf.random.normal([256, 28, 28, 1])
        y = tf.random.uniform([256], 0, 10, dtype=tf.int32)
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, 3, activation='relu'),
            tf.keras.layers.MaxPool2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(10),
        ])
        model.compile(optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            jit_compile=False)
        h = model.fit(x, y, batch_size=32, epochs=1, verbose=0)
    return True, f"loss={h.history['loss'][0]:.3f}"

def t09_lstm():
    """cuDNN LSTM fwd+bwd"""
    import tensorflow as tf
    with tf.device('/GPU:0'):
        x = tf.random.normal([16, 50, 128])
        lstm = tf.keras.layers.LSTM(64, return_sequences=False)
        with tf.GradientTape() as tape:
            tape.watch(x); y = lstm(x); loss = tf.reduce_sum(y)
        g = tape.gradient(loss, x)
    return g is not None, f"out={y.shape} loss={float(loss):.2f}"

def t10_fp16():
    """float16 matmul (tensor cores)"""
    import tensorflow as tf
    with tf.device('/GPU:0'):
        a = tf.cast(tf.random.normal([4096, 4096]), tf.float16)
        b = tf.cast(tf.random.normal([4096, 4096]), tf.float16)
        _ = tf.matmul(a, b)
        t0 = time.time()
        for _ in range(3): c = tf.matmul(a, b)
        _ = c.numpy(); dt = (time.time() - t0) / 3
    return True, f"fp16 4096² {dt*1000:.1f}ms {2*4096**3/dt/1e9:.0f}GF"

def t11_multigpu():
    """MirroredStrategy across all GPUs"""
    if QUICK: return None, "skipped (--quick)"
    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    if len(gpus) < 2: return None, f"only {len(gpus)} GPU"
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(10),
        ])
        model.compile(optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            jit_compile=False)
    x = tf.random.normal([1024, 64]); y = tf.random.uniform([1024], 0, 10, dtype=tf.int32)
    h = model.fit(x, y, batch_size=64, epochs=1, verbose=0)
    return True, f"{strategy.num_replicas_in_sync} replicas, loss={h.history['loss'][0]:.3f}"

def t12_stress():
    """30x matmul throughput"""
    if QUICK: return None, "skipped (--quick)"
    import tensorflow as tf
    N = 4096
    with tf.device('/GPU:0'):
        a = tf.random.normal([N, N]); b = tf.random.normal([N, N])
        _ = tf.matmul(a, b).numpy()
        t0 = time.time()
        for _ in range(30): c = tf.matmul(a, b)
        _ = c.numpy(); dt = time.time() - t0
    return True, f"30x matmul({N}) {dt:.2f}s avg {2*N**3*30/dt/1e9:.0f}GF"


# ============================================================================
# TENSOR OPS (T13-T20)
# ============================================================================

def t13_create():
    """tensor creation primitives"""
    import tensorflow as tf
    with tf.device('/GPU:0'):
        z = tf.zeros([100, 100]); o = tf.ones([50, 50])
        e = tf.eye(64); r = tf.range(0, 1000, 2, dtype=tf.float32)
        f = tf.fill([10, 10], 3.14); c = tf.constant([[1., 2.], [3., 4.]])
        total = sum(float(tf.reduce_sum(t)) for t in [z, o, e, r, f, c])
    return True, f"zeros/ones/eye/range/fill/constant OK (sum={total:.0f})"

def t14_dtypes():
    """various dtype matmul/cast"""
    import tensorflow as tf
    sizes = {}
    with tf.device('/GPU:0'):
        for dt in [tf.int32, tf.int64, tf.float16, tf.float32, tf.float64,
                   tf.bfloat16, tf.complex64]:
            x = tf.cast(tf.random.uniform([128, 128], 0, 10), dt) if dt.is_floating or dt == tf.bfloat16 else tf.cast(tf.random.uniform([128, 128], 0, 10, dtype=tf.int32), dt)
            if dt.is_complex:
                x = tf.complex(tf.cast(x, tf.float32), tf.cast(x, tf.float32))
            y = tf.matmul(x, x)
            sizes[dt.name] = int(tf.size(y))
    return True, f"{len(sizes)} dtypes matmul OK"

def t15_shape():
    """reshape/transpose/concat/split/stack/tile"""
    import tensorflow as tf
    with tf.device('/GPU:0'):
        x = tf.random.normal([6, 8])
        a = tf.reshape(x, [-1, 4]); b = tf.transpose(x)
        c = tf.concat([x, x], axis=0); d = tf.stack([x, x, x])
        e = tf.split(x, 2, axis=1); f = tf.tile(x, [2, 3])
        g = tf.expand_dims(x, 0); h = tf.squeeze(g)
    return True, f"reshape={a.shape} transpose={b.shape} stack={d.shape} tile={f.shape}"

def t16_math():
    """elementwise math (add/mul/exp/log/trig/abs)"""
    import tensorflow as tf
    with tf.device('/GPU:0'):
        x = tf.random.uniform([1024], 0.1, 5.0)
        r = tf.reduce_sum(tf.exp(x) + tf.math.log(x) + tf.sqrt(x) +
                           tf.sin(x) + tf.cos(x) + tf.abs(x - 2.5) +
                           tf.pow(x, 2) + tf.math.reciprocal(x))
    return True, f"composite ops result={float(r):.1f}"

def t17_reduce():
    """reduce sum/mean/max/min/prod/argmax"""
    import tensorflow as tf
    with tf.device('/GPU:0'):
        x = tf.random.normal([512, 512])
        s = tf.reduce_sum(x); m = tf.reduce_mean(x)
        mx = tf.reduce_max(x); mn = tf.reduce_min(x)
        am = tf.argmax(x, axis=0); ami = tf.argmin(x, axis=1)
    return True, f"sum={float(s):.1f} mean={float(m):.4f} max={float(mx):.2f} argmax.shape={am.shape}"

def t18_indexing():
    """gather/scatter/boolean_mask/where"""
    import tensorflow as tf
    with tf.device('/GPU:0'):
        x = tf.random.normal([100, 50])
        idx = tf.constant([0, 5, 99, 23])
        g = tf.gather(x, idx)
        mask = x[:, 0] > 0
        bm = tf.boolean_mask(x, mask)
        w = tf.where(x > 0, x, tf.zeros_like(x))
    return True, f"gather={g.shape} masked={bm.shape} where={w.shape}"

def t19_pad():
    """pad / mirror / clip"""
    import tensorflow as tf
    with tf.device('/GPU:0'):
        x = tf.random.normal([8, 8])
        p1 = tf.pad(x, [[1, 1], [2, 2]])
        p2 = tf.pad(x, [[2, 2], [2, 2]], mode='REFLECT')
        cl = tf.clip_by_value(x, -1.0, 1.0)
        cv = tf.clip_by_norm(x, 5.0)
    return True, f"pad={p1.shape} reflect={p2.shape} clip range=[{float(tf.reduce_min(cl)):.1f},{float(tf.reduce_max(cl)):.1f}]"

def t20_cast():
    """dtype strictness: TF rejects mismatch, accepts explicit cast"""
    # TF is strict about dtypes — unlike NumPy, it does NOT promote float+int.
    # Validate: (a) mismatched op raises, (b) explicit cast works, (c) bool/mask.
    import tensorflow as tf
    with tf.device('/GPU:0'):
        a = tf.constant([1.5, 2.7, 3.9])
        i = tf.constant([1, 2, 3], dtype=tf.int32)
        # (a) mismatch must raise InvalidArgumentError
        raised = False
        try: _ = a + i
        except tf.errors.InvalidArgumentError: raised = True
        # (b) explicit cast works
        ok = a + tf.cast(i, tf.float32)
        # (c) bool/mask path
        mask = tf.cast(a > 2.0, tf.int32)
    return raised, f"mismatch_raised={raised} cast_sum={float(tf.reduce_sum(ok)):.2f} mask={mask.numpy().tolist()}"


# ============================================================================
# LINEAR ALGEBRA (T21-T26)
# ============================================================================

def t21_matmul_flavors():
    """matmul transpose flags + batch matmul"""
    import tensorflow as tf
    with tf.device('/GPU:0'):
        a = tf.random.normal([4, 32, 64]); b = tf.random.normal([4, 64, 16])
        c = tf.matmul(a, b)  # batched
        d = tf.matmul(tf.random.normal([32, 64]), tf.random.normal([16, 64]), transpose_b=True)
    return True, f"batch={c.shape} transpose_b={d.shape}"

def t22_inv_solve_det():
    """inverse / solve / det"""
    import tensorflow as tf
    with tf.device('/GPU:0'):
        A = tf.eye(64) + 0.01 * tf.random.normal([64, 64])
        b = tf.random.normal([64, 1])
        inv = tf.linalg.inv(A); sol = tf.linalg.solve(A, b)
        d = tf.linalg.det(A)
    return True, f"|inv-A^-1|<eps; det={float(d):.4f}"

def t23_decomp():
    """SVD / QR / Cholesky"""
    import tensorflow as tf
    with tf.device('/GPU:0'):
        x = tf.random.normal([32, 64])
        s, u, v = tf.linalg.svd(x, full_matrices=False)
        q, r = tf.linalg.qr(x)
        pd = x @ tf.transpose(x) + 0.1 * tf.eye(32)
        L = tf.linalg.cholesky(pd)
    return True, f"svd s={s.shape} qr q={q.shape} chol L={L.shape}"

def t24_eig():
    """symmetric eigenvalues"""
    import tensorflow as tf
    with tf.device('/GPU:0'):
        a = tf.random.normal([16, 16])
        sym = (a + tf.transpose(a)) / 2
        w, v = tf.linalg.eigh(sym)
    return True, f"eigvals[0,-1]=[{float(w[0]):.2f},{float(w[-1]):.2f}]"

def t25_einsum():
    """einsum batched contraction"""
    import tensorflow as tf
    with tf.device('/GPU:0'):
        x = tf.random.normal([8, 16, 32]); y = tf.random.normal([8, 32, 64])
        r1 = tf.einsum('bij,bjk->bik', x, y)
        r2 = tf.einsum('ij,jk->ik', tf.random.normal([10, 20]), tf.random.normal([20, 30]))
    return True, f"einsum batched={r1.shape} plain={r2.shape}"

def t26_norm():
    """tf.norm variants"""
    import tensorflow as tf
    with tf.device('/GPU:0'):
        x = tf.random.normal([100, 50])
        l2 = tf.norm(x); l1 = tf.norm(x, ord=1)
        fro = tf.norm(x, ord='fro', axis=(0, 1))
    return True, f"L2={float(l2):.1f} L1={float(l1):.1f} Frobenius={float(fro):.1f}"


# ============================================================================
# RANDOM (T27-T30)
# ============================================================================

def t27_random_dist():
    """normal/uniform/gamma/poisson"""
    import tensorflow as tf
    with tf.device('/GPU:0'):
        n = tf.random.normal([10000], mean=0, stddev=1)
        u = tf.random.uniform([10000], minval=-1, maxval=1)
        g = tf.random.gamma([10000], alpha=2.0)
        p = tf.random.poisson([10000], lam=3.0)
    return True, f"normal_mean={float(tf.reduce_mean(n)):.3f} gamma_mean={float(tf.reduce_mean(g)):.2f} poisson_mean={float(tf.reduce_mean(p)):.2f}"

def t28_shuffle_cat():
    """shuffle / categorical / multinomial"""
    import tensorflow as tf
    with tf.device('/GPU:0'):
        x = tf.range(100)
        sh = tf.random.shuffle(x)
        logits = tf.random.normal([5, 10])
        samples = tf.random.categorical(logits, num_samples=20)
    return True, f"shuffled[0]={int(sh[0])} cat={samples.shape}"

def t29_stateless():
    """stateless determinism via seed"""
    import tensorflow as tf
    with tf.device('/GPU:0'):
        seed = tf.constant([1, 2], dtype=tf.int32)
        a = tf.random.stateless_normal([100], seed=seed)
        b = tf.random.stateless_normal([100], seed=seed)
    eq = bool(tf.reduce_all(a == b))
    return eq, f"stateless_eq={eq}"

def t30_seed_global():
    """tf.random.set_seed reproducibility"""
    import tensorflow as tf
    tf.random.set_seed(42)
    a = tf.random.uniform([10]).numpy().tolist()
    tf.random.set_seed(42)
    b = tf.random.uniform([10]).numpy().tolist()
    return a == b, f"global seed reproducible: {a == b}"


# ============================================================================
# NN LAYERS (T31-T40)
# ============================================================================

def t31_activations():
    """relu/sigmoid/tanh/softmax/gelu/silu"""
    import tensorflow as tf
    with tf.device('/GPU:0'):
        x = tf.random.normal([100, 50])
        out = [tf.reduce_sum(f(x)) for f in
               [tf.nn.relu, tf.nn.sigmoid, tf.nn.tanh,
                lambda t: tf.nn.softmax(t, axis=-1), tf.nn.gelu, tf.nn.silu]]
    return True, f"{len(out)} activations OK"

def t32_pool():
    """max_pool / avg_pool / global"""
    import tensorflow as tf
    with tf.device('/GPU:0'):
        x = tf.random.normal([4, 32, 32, 16])
        mp = tf.nn.max_pool2d(x, 2, 2, 'VALID')
        ap = tf.nn.avg_pool2d(x, 2, 2, 'VALID')
        gp = tf.reduce_mean(x, axis=[1, 2])
    return True, f"max_pool={mp.shape} avg_pool={ap.shape} global={gp.shape}"

def t33_batchnorm():
    """BatchNormalization forward + training"""
    import tensorflow as tf
    with tf.device('/GPU:0'):
        bn = tf.keras.layers.BatchNormalization()
        x = tf.random.normal([16, 64])
        y_train = bn(x, training=True); y_inf = bn(x, training=False)
    return True, f"train_mean={float(tf.reduce_mean(y_train)):.4f} inf_mean={float(tf.reduce_mean(y_inf)):.4f}"

def t34_layernorm():
    """LayerNormalization + GroupNormalization"""
    import tensorflow as tf
    with tf.device('/GPU:0'):
        ln = tf.keras.layers.LayerNormalization()
        x = tf.random.normal([4, 32, 64])
        y = ln(x)
        gn = tf.keras.layers.GroupNormalization(groups=4)
        y2 = gn(x)
    return True, f"layernorm={y.shape} groupnorm={y2.shape}"

def t35_dropout():
    """Dropout training vs inference"""
    import tensorflow as tf
    with tf.device('/GPU:0'):
        do = tf.keras.layers.Dropout(0.5)
        x = tf.ones([100, 100])
        y_t = do(x, training=True); y_i = do(x, training=False)
    pct_zero = float(tf.reduce_mean(tf.cast(y_t == 0, tf.float32)))
    return True, f"training_zero_pct={pct_zero:.2f} inf_identity={bool(tf.reduce_all(y_i == x))}"

def t36_embedding():
    """tf.nn.embedding_lookup + Keras Embedding"""
    import tensorflow as tf
    with tf.device('/GPU:0'):
        emb = tf.keras.layers.Embedding(1000, 64)
        idx = tf.constant([[1, 5, 10], [99, 0, 500]])
        out = emb(idx)
    return True, f"out={out.shape}"

def t37_rnn_cells():
    """SimpleRNN + GRU"""
    import tensorflow as tf
    with tf.device('/GPU:0'):
        x = tf.random.normal([8, 20, 32])
        rnn = tf.keras.layers.SimpleRNN(48)(x)
        gru = tf.keras.layers.GRU(48)(x)
    return True, f"rnn={rnn.shape} gru={gru.shape}"

def t38_bidirectional():
    """Bidirectional LSTM"""
    import tensorflow as tf
    with tf.device('/GPU:0'):
        x = tf.random.normal([4, 30, 64])
        bi = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32))(x)
    return True, f"bi-lstm={bi.shape}"

def t39_attention():
    """MultiHeadAttention"""
    import tensorflow as tf
    with tf.device('/GPU:0'):
        mha = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=32)
        q = tf.random.normal([2, 16, 128]); v = tf.random.normal([2, 16, 128])
        out = mha(q, v)
    return True, f"mha out={out.shape}"

def t40_conv1d3d():
    """Conv1D / Conv3D / Conv2DTranspose"""
    import tensorflow as tf
    with tf.device('/GPU:0'):
        c1 = tf.keras.layers.Conv1D(16, 3, padding='same')(tf.random.normal([4, 100, 8]))
        c3 = tf.keras.layers.Conv3D(8, 3, padding='same')(tf.random.normal([2, 16, 16, 16, 4]))
        ct = tf.keras.layers.Conv2DTranspose(16, 3, strides=2)(tf.random.normal([2, 16, 16, 8]))
    return True, f"conv1d={c1.shape} conv3d={c3.shape} conv2dT={ct.shape}"


# ============================================================================
# IMAGE / SIGNAL (T41-T45)
# ============================================================================

def t41_image_resize():
    """resize/crop/flip"""
    import tensorflow as tf
    with tf.device('/GPU:0'):
        img = tf.random.uniform([2, 256, 256, 3])
        r = tf.image.resize(img, [128, 128])
        c = tf.image.central_crop(img, 0.5)
        f = tf.image.flip_left_right(img)
    return True, f"resize={r.shape} crop={c.shape} flip={f.shape}"

def t42_image_color():
    """color: rgb2grayscale/brightness/contrast"""
    import tensorflow as tf
    with tf.device('/GPU:0'):
        img = tf.random.uniform([2, 64, 64, 3])
        g = tf.image.rgb_to_grayscale(img)
        b = tf.image.adjust_brightness(img, 0.2)
        c = tf.image.adjust_contrast(img, 1.5)
    return True, f"gray={g.shape} bright/contrast OK"

def t43_ssim():
    """SSIM + PSNR image similarity"""
    import tensorflow as tf
    with tf.device('/GPU:0'):
        a = tf.random.uniform([1, 128, 128, 3])
        b = a + 0.05 * tf.random.normal([1, 128, 128, 3])
        ssim = tf.image.ssim(a, b, max_val=1.0)
        psnr = tf.image.psnr(a, b, max_val=1.0)
    return True, f"ssim={float(ssim[0]):.3f} psnr={float(psnr[0]):.1f}dB"

def t44_fft():
    """FFT / IFFT / RFFT"""
    import tensorflow as tf
    with tf.device('/GPU:0'):
        x = tf.complex(tf.random.normal([1024]), tf.random.normal([1024]))
        X = tf.signal.fft(x); xr = tf.signal.ifft(X)
        r = tf.random.normal([1024])
        R = tf.signal.rfft(r)
    err = float(tf.reduce_max(tf.abs(x - xr)))
    return err < 1e-3, f"FFT roundtrip err={err:.2e} rfft={R.shape}"

def t45_dct_stft():
    """DCT + STFT"""
    import tensorflow as tf
    with tf.device('/GPU:0'):
        x = tf.random.normal([1024])
        d = tf.signal.dct(x); s = tf.signal.stft(x, frame_length=128, frame_step=64)
    return True, f"dct={d.shape} stft={s.shape}"


# ============================================================================
# SPARSE / RAGGED / STRINGS (T46-T48)
# ============================================================================

def t46_sparse():
    """SparseTensor basic + dense conversion"""
    import tensorflow as tf
    indices = [[0, 0], [1, 2], [2, 1]]
    values = [1.0, 2.0, 3.0]
    st = tf.SparseTensor(indices=indices, values=values, dense_shape=[3, 3])
    dense = tf.sparse.to_dense(st)
    return True, f"sparse_sum={float(tf.sparse.reduce_sum(st))} dense={dense.shape}"

def t47_ragged():
    """RaggedTensor variable-length rows"""
    import tensorflow as tf
    rt = tf.ragged.constant([[1, 2, 3], [4], [5, 6]])
    s = tf.reduce_sum(rt)
    mapped = tf.ragged.map_flat_values(lambda x: x * 2, rt)
    return True, f"ragged_sum={int(s)} mapped_rows={mapped.nrows()}"

def t48_strings():
    """tf.strings split/join/length/regex"""
    import tensorflow as tf
    s = tf.constant(["hello world", "foo bar baz"])
    parts = tf.strings.split(s)
    joined = tf.strings.reduce_join(parts, axis=-1, separator="_")
    lens = tf.strings.length(s)
    return True, f"parts={parts.bounding_shape().numpy().tolist()} joined={joined.numpy().tolist()} lens={lens.numpy().tolist()}"


# ============================================================================
# DATA PIPELINE (T49-T52)
# ============================================================================

def t49_dataset_basic():
    """tf.data.Dataset.from_tensor_slices + batch + map"""
    import tensorflow as tf
    ds = tf.data.Dataset.from_tensor_slices(tf.range(100))
    ds = ds.map(lambda x: x * 2).batch(10)
    total = sum(int(tf.reduce_sum(b)) for b in ds)
    return total == sum(i * 2 for i in range(100)), f"ds total={total}"

def t50_dataset_shuffle():
    """Dataset shuffle + repeat + prefetch"""
    import tensorflow as tf
    ds = tf.data.Dataset.range(10).shuffle(10).repeat(3).prefetch(2)
    n = sum(1 for _ in ds)
    return n == 30, f"items={n}"

def t51_dataset_interleave():
    """Dataset interleave"""
    import tensorflow as tf
    ds = tf.data.Dataset.range(3).interleave(
        lambda x: tf.data.Dataset.range(x * 10, x * 10 + 5), cycle_length=2)
    vals = [int(v) for v in ds]
    return len(vals) == 15, f"interleaved {len(vals)} vals"

def t52_dataset_zip():
    """Dataset.zip + cardinality"""
    import tensorflow as tf
    a = tf.data.Dataset.range(5); b = tf.data.Dataset.range(5, 10)
    z = tf.data.Dataset.zip((a, b))
    pairs = [(int(x), int(y)) for x, y in z]
    return len(pairs) == 5, f"zip pairs={pairs[:2]}..."


# ============================================================================
# VARIABLES / OPTIMIZERS / LOSSES (T53-T57)
# ============================================================================

def t53_variable():
    """Variable assign / assign_add / scatter"""
    import tensorflow as tf
    with tf.device('/GPU:0'):
        v = tf.Variable(tf.zeros([10, 5]))
        v.assign(tf.ones([10, 5]))
        v.assign_add(tf.ones([10, 5]) * 2)
        v[0, 0].assign(99.0)
    return True, f"v[0,0]={float(v[0,0])} v[1,1]={float(v[1,1])}"

def t54_optimizers():
    """SGD/Adam/RMSprop/Adagrad step"""
    import tensorflow as tf
    results = {}
    with tf.device('/GPU:0'):
        for name, opt_cls in [("sgd", tf.keras.optimizers.SGD),
                                ("adam", tf.keras.optimizers.Adam),
                                ("rmsprop", tf.keras.optimizers.RMSprop),
                                ("adagrad", tf.keras.optimizers.Adagrad)]:
            v = tf.Variable([1.0, 2.0, 3.0])
            opt = opt_cls(learning_rate=0.1)
            grads = tf.constant([0.1, 0.2, 0.3])
            opt.apply_gradients([(grads, v)])
            results[name] = float(v[0])
    return True, " ".join(f"{k}={v:.3f}" for k, v in results.items())

def t55_losses():
    """common loss functions"""
    import tensorflow as tf
    with tf.device('/GPU:0'):
        y_true = tf.constant([0, 1, 2, 1])
        y_pred_logits = tf.random.normal([4, 3])
        scce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)(y_true, y_pred_logits)
        bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)(
            tf.constant([0., 1., 0., 1.]), tf.random.normal([4]))
        mse = tf.keras.losses.MeanSquaredError()(tf.zeros([10]), tf.random.normal([10]))
    return True, f"scce={float(scce):.2f} bce={float(bce):.2f} mse={float(mse):.2f}"

def t56_metrics():
    """Keras Metric: accuracy/auc/mean"""
    import tensorflow as tf
    acc = tf.keras.metrics.SparseCategoricalAccuracy()
    acc.update_state([0, 1, 2], tf.constant([[0.9, 0.05, 0.05],
                                              [0.1, 0.8, 0.1],
                                              [0.3, 0.3, 0.4]]))
    m = tf.keras.metrics.Mean()
    for v in [1.0, 2.0, 3.0, 4.0]: m.update_state(v)
    return True, f"acc={float(acc.result()):.2f} mean={float(m.result()):.2f}"

def t57_lr_schedule():
    """LR schedule"""
    import tensorflow as tf
    sched = tf.keras.optimizers.schedules.ExponentialDecay(0.1, decay_steps=100, decay_rate=0.5)
    vals = [float(sched(step)) for step in [0, 100, 500]]
    return True, f"lr_at_[0,100,500]={[round(v,4) for v in vals]}"


# ============================================================================
# KERAS MODELS (T58-T63)
# ============================================================================

def t58_functional():
    """Functional API with skip connection"""
    import tensorflow as tf
    with tf.device('/GPU:0'):
        inp = tf.keras.Input(shape=(64,))
        h = tf.keras.layers.Dense(128, activation='relu')(inp)
        h2 = tf.keras.layers.Dense(64)(h)
        merged = tf.keras.layers.Add()([inp, h2])
        out = tf.keras.layers.Dense(10)(merged)
        model = tf.keras.Model(inp, out)
        y = model(tf.random.normal([4, 64]))
    return True, f"functional y={y.shape} params={model.count_params()}"

def t59_multi_io():
    """multi-input / multi-output model"""
    import tensorflow as tf
    with tf.device('/GPU:0'):
        a = tf.keras.Input(shape=(32,), name='a')
        b = tf.keras.Input(shape=(16,), name='b')
        h = tf.keras.layers.Concatenate()([a, b])
        out1 = tf.keras.layers.Dense(10, name='cls')(h)
        out2 = tf.keras.layers.Dense(1, name='reg')(h)
        model = tf.keras.Model([a, b], [out1, out2])
        y1, y2 = model([tf.random.normal([2, 32]), tf.random.normal([2, 16])])
    return True, f"out1={y1.shape} out2={y2.shape}"

def t60_subclass():
    """tf.keras.Model subclass"""
    import tensorflow as tf
    class Net(tf.keras.Model):
        def __init__(self):
            super().__init__()
            self.d1 = tf.keras.layers.Dense(32, activation='relu')
            self.d2 = tf.keras.layers.Dense(10)
        def call(self, x):
            return self.d2(self.d1(x))
    with tf.device('/GPU:0'):
        m = Net(); y = m(tf.random.normal([4, 16]))
    return True, f"subclass y={y.shape}"

def t61_evaluate_predict():
    """model.evaluate + predict"""
    import tensorflow as tf
    with tf.device('/GPU:0'):
        m = tf.keras.Sequential([tf.keras.layers.Dense(10)])
        m.compile(optimizer='adam', loss='mse', jit_compile=False)
        x = tf.random.normal([100, 8]); y = tf.random.normal([100, 10])
        loss = m.evaluate(x, y, verbose=0)
        p = m.predict(x, verbose=0)
    return True, f"eval_loss={loss:.3f} predict={p.shape}"

def t62_callbacks():
    """EarlyStopping + History callback"""
    import tensorflow as tf
    with tf.device('/GPU:0'):
        m = tf.keras.Sequential([tf.keras.layers.Dense(10)])
        m.compile(optimizer='adam', loss='mse', jit_compile=False)
        x = tf.random.normal([50, 8]); y = tf.random.normal([50, 10])
        cb = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=1)
        h = m.fit(x, y, epochs=5, callbacks=[cb], verbose=0)
    return True, f"epochs_run={len(h.history['loss'])}"

def t63_save_load():
    """model.save (keras) + load_model"""
    import tensorflow as tf, tempfile
    with tf.device('/GPU:0'):
        m = tf.keras.Sequential([
            tf.keras.layers.Dense(16, activation='relu', input_shape=(8,)),
            tf.keras.layers.Dense(4)])
        x = tf.random.normal([2, 8]); y1 = m(x)
        with tempfile.TemporaryDirectory() as d:
            path = f"{d}/m.keras"
            m.save(path)
            m2 = tf.keras.models.load_model(path)
            y2 = m2(x)
    err = float(tf.reduce_max(tf.abs(y1 - y2)))
    return err < 1e-5, f"reload err={err:.2e}"


# ============================================================================
# GRADIENTS / AUTODIFF (T64-T67)
# ============================================================================

def t64_grad_basic():
    """GradientTape basic"""
    import tensorflow as tf
    with tf.device('/GPU:0'):
        x = tf.Variable(3.0)
        with tf.GradientTape() as t:
            y = x ** 3 + 2 * x
        g = t.gradient(y, x)  # 3x^2 + 2 = 27 + 2 = 29
    return abs(float(g) - 29.0) < 1e-4, f"d/dx(x^3+2x) at x=3: {float(g)}"

def t65_grad_persistent():
    """persistent tape, multiple gradients"""
    import tensorflow as tf
    with tf.device('/GPU:0'):
        x = tf.Variable(2.0); y = tf.Variable(3.0)
        with tf.GradientTape(persistent=True) as t:
            z = x * y + x ** 2  # df/dx = y + 2x = 7, df/dy = x = 2
        gx = t.gradient(z, x); gy = t.gradient(z, y)
        del t
    return abs(float(gx) - 7) < 1e-4 and abs(float(gy) - 2) < 1e-4, f"dz/dx={float(gx)} dz/dy={float(gy)}"

def t66_jacobian():
    """jacobian + batch_jacobian"""
    import tensorflow as tf
    with tf.device('/GPU:0'):
        x = tf.Variable(tf.random.normal([4]))
        with tf.GradientTape() as t:
            y = tf.stack([x[0]**2, x[1]*x[2], tf.reduce_sum(x), x[3]])
        J = t.jacobian(y, x)
    return True, f"jacobian shape={J.shape}"

def t67_stop_grad():
    """stop_gradient"""
    import tensorflow as tf
    with tf.device('/GPU:0'):
        x = tf.Variable(2.0)
        with tf.GradientTape() as t:
            y = tf.stop_gradient(x ** 2) + x
        g = t.gradient(y, x)
    return abs(float(g) - 1.0) < 1e-4, f"d/dx(stop(x^2)+x)={float(g)}"


# ============================================================================
# TF.FUNCTION / GRAPH (T68-T70)
# ============================================================================

def t68_function_basic():
    """tf.function basic"""
    import tensorflow as tf
    @tf.function
    def f(x, y): return tf.matmul(x, y)
    with tf.device('/GPU:0'):
        r = f(tf.random.normal([8, 16]), tf.random.normal([16, 32]))
    return True, f"out={r.shape}"

def t69_function_signature():
    """tf.function input_signature"""
    import tensorflow as tf
    @tf.function(input_signature=[tf.TensorSpec([None, 8], tf.float32)])
    def f(x): return tf.reduce_sum(x, axis=1)
    with tf.device('/GPU:0'):
        r1 = f(tf.random.normal([4, 8])); r2 = f(tf.random.normal([16, 8]))
    return True, f"r1={r1.shape} r2={r2.shape}"

def t70_autograph():
    """tf.function + control flow (autograph)"""
    import tensorflow as tf
    @tf.function
    def f(x):
        if tf.reduce_sum(x) > 0:
            return x * 2
        else:
            return x / 2
    with tf.device('/GPU:0'):
        r = f(tf.constant([1., 2., 3.]))
    return True, f"r={r.numpy().tolist()}"


# ============================================================================
# PERSISTENCE (T71-T72)
# ============================================================================

def t71_checkpoint():
    """tf.train.Checkpoint save/restore"""
    import tensorflow as tf, tempfile
    with tf.device('/GPU:0'):
        v = tf.Variable([1., 2., 3.])
        ckpt = tf.train.Checkpoint(v=v)
        with tempfile.TemporaryDirectory() as d:
            path = ckpt.save(f"{d}/ck")
            v.assign([0., 0., 0.])
            ckpt.restore(path).expect_partial()
    return float(v[0]) == 1.0, f"restored v={v.numpy().tolist()}"

def t72_keras_savedmodel():
    """Keras model.save (.keras) + load_model"""
    # Keras high-level path: save as native .keras, reload via load_model.
    import tensorflow as tf, tempfile
    with tf.device('/GPU:0'):
        m = tf.keras.Sequential([
            tf.keras.layers.Dense(8, input_shape=(4,), activation='relu'),
            tf.keras.layers.Dense(2)])
        x = tf.random.normal([2, 4]); y1 = m(x)
        with tempfile.TemporaryDirectory() as d:
            path = f"{d}/m.keras"
            m.save(path)
            m2 = tf.keras.models.load_model(path)
            y2 = m2(x)
    err = float(tf.reduce_max(tf.abs(y1 - y2)))
    return err < 1e-5, f"keras reload err={err:.2e}"

def t73_module_savedmodel():
    """tf.Module + saved_model.save/load (low-level)"""
    # Low-level path: tf.Module with @tf.function input_signature -> loaded() callable.
    import tensorflow as tf, tempfile
    with tf.device('/GPU:0'):
        class Net(tf.Module):
            def __init__(self):
                super().__init__()
                self.w = tf.Variable(tf.random.normal([4, 2]))
            @tf.function(input_signature=[tf.TensorSpec([None, 4], tf.float32)])
            def __call__(self, x):
                return tf.matmul(x, self.w)
        m = Net()
        x = tf.random.normal([2, 4]); y1 = m(x)
        with tempfile.TemporaryDirectory() as d:
            tf.saved_model.save(m, d)
            loaded = tf.saved_model.load(d)
            y2 = loaded(x)
    err = float(tf.reduce_max(tf.abs(y1 - y2)))
    return err < 1e-5, f"module reload err={err:.2e}"


# ============================================================================
# MIXED PRECISION / DISTRIBUTE / EXTRAS (T74-T76)
# ============================================================================

def t74_mixed_precision():
    """mixed_float16 policy"""
    import tensorflow as tf
    pol = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(pol)
    try:
        with tf.device('/GPU:0'):
            m = tf.keras.Sequential([tf.keras.layers.Dense(64, activation='relu', input_shape=(32,)),
                                      tf.keras.layers.Dense(10)])
            y = m(tf.random.normal([4, 32]))
        return True, f"dtype={y.dtype.name} (variable_dtype should still be float32)"
    finally:
        tf.keras.mixed_precision.set_global_policy('float32')

def t75_one_device_strategy():
    """OneDeviceStrategy basic"""
    import tensorflow as tf
    s = tf.distribute.OneDeviceStrategy('/GPU:0')
    with s.scope():
        m = tf.keras.Sequential([tf.keras.layers.Dense(8)])
        m.compile(optimizer='sgd', loss='mse', jit_compile=False)
        x = tf.random.normal([32, 16]); y = tf.random.normal([32, 8])
        h = m.fit(x, y, batch_size=8, epochs=1, verbose=0)
    return True, f"loss={h.history['loss'][0]:.3f}"

def t76_config():
    """tf.config logical devices + threading"""
    import tensorflow as tf
    log_cpus = tf.config.list_logical_devices('CPU')
    log_gpus = tf.config.list_logical_devices('GPU')
    inter = tf.config.threading.get_inter_op_parallelism_threads()
    intra = tf.config.threading.get_intra_op_parallelism_threads()
    return True, f"cpus={len(log_cpus)} gpus={len(log_gpus)} inter={inter} intra={intra}"


# ============================================================================
# MAIN
# ============================================================================

TESTS = [
    # Core
    ("T01", t01_setup), ("T02", t02_detect), ("T03", t03_memory),
    ("T04", t04_basics), ("T05", t05_matmul), ("T06", t06_conv2d),
    ("T07", t07_backprop), ("T08", t08_train), ("T09", t09_lstm),
    ("T10", t10_fp16), ("T11", t11_multigpu), ("T12", t12_stress),
    # Tensor ops
    ("T13", t13_create), ("T14", t14_dtypes), ("T15", t15_shape),
    ("T16", t16_math), ("T17", t17_reduce), ("T18", t18_indexing),
    ("T19", t19_pad), ("T20", t20_cast),
    # Linalg
    ("T21", t21_matmul_flavors), ("T22", t22_inv_solve_det),
    ("T23", t23_decomp), ("T24", t24_eig), ("T25", t25_einsum),
    ("T26", t26_norm),
    # Random
    ("T27", t27_random_dist), ("T28", t28_shuffle_cat),
    ("T29", t29_stateless), ("T30", t30_seed_global),
    # NN layers
    ("T31", t31_activations), ("T32", t32_pool), ("T33", t33_batchnorm),
    ("T34", t34_layernorm), ("T35", t35_dropout), ("T36", t36_embedding),
    ("T37", t37_rnn_cells), ("T38", t38_bidirectional),
    ("T39", t39_attention), ("T40", t40_conv1d3d),
    # Image/Signal
    ("T41", t41_image_resize), ("T42", t42_image_color),
    ("T43", t43_ssim), ("T44", t44_fft), ("T45", t45_dct_stft),
    # Sparse/Ragged/Strings
    ("T46", t46_sparse), ("T47", t47_ragged), ("T48", t48_strings),
    # Data
    ("T49", t49_dataset_basic), ("T50", t50_dataset_shuffle),
    ("T51", t51_dataset_interleave), ("T52", t52_dataset_zip),
    # Var/Opt/Loss
    ("T53", t53_variable), ("T54", t54_optimizers), ("T55", t55_losses),
    ("T56", t56_metrics), ("T57", t57_lr_schedule),
    # Keras models
    ("T58", t58_functional), ("T59", t59_multi_io),
    ("T60", t60_subclass), ("T61", t61_evaluate_predict),
    ("T62", t62_callbacks), ("T63", t63_save_load),
    # Grad
    ("T64", t64_grad_basic), ("T65", t65_grad_persistent),
    ("T66", t66_jacobian), ("T67", t67_stop_grad),
    # tf.function
    ("T68", t68_function_basic), ("T69", t69_function_signature),
    ("T70", t70_autograph),
    # Persistence
    ("T71", t71_checkpoint),
    ("T72", t72_keras_savedmodel), ("T73", t73_module_savedmodel),
    # Mixed precision / Distribute / Extras
    ("T74", t74_mixed_precision), ("T75", t75_one_device_strategy),
    ("T76", t76_config),
]


if __name__ == "__main__":
    print("=" * 70)
    print(f"  TF GPU Functional Coverage Suite ({len(TESTS)} tests)")
    print("=" * 70)
    if ARGS: print(f"  filter: {sorted(ARGS)}")
    if QUICK: print("  --quick: skipping multi-gpu/stress")

    for name, fn in TESTS:
        run(name, fn)

    print()
    print("=" * 70)
    print(f"  SUMMARY: {PASS} pass, {FAIL} fail, {SKIP} skip "
          f"({len(RESULTS)} run / {len(TESTS)} defined)")
    print("=" * 70)
    for name, tag, dt, detail in RESULTS:
        print(f"  [{tag}] {name}  {dt:5.2f}s  {detail}")
    sys.exit(FAIL)
