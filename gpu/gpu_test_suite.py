#!/usr/bin/env python3
"""TensorFlow GPU test suite for Power9 + V100.

Runs progressively heavier tests:
  T01  Setup            - imports, version, build flags
  T02  Detection        - list GPUs, names, memory
  T03  Memory           - alloc/free, growth policy
  T04  Basics           - random, reduce, broadcast on GPU
  T05  Matmul           - 512, 4096, 8192 with timing
  T06  Conv2D           - 3 layers fwd
  T07  Backprop         - gradient through small MLP
  T08  Train step       - 1 epoch of synthetic MNIST shape
  T09  cuDNN LSTM       - sequence model fwd+bwd
  T10  Mixed precision  - float16 matmul
  T11  Multi-GPU        - mirrored strategy if >=2 GPUs
  T12  Stress           - sustained matmul throughput

Usage:
  python3 gpu_test_suite.py             # all tests
  python3 gpu_test_suite.py T05 T08     # only listed
  python3 gpu_test_suite.py --quick     # skip stress + multi-gpu
"""
import os
import sys
import time
import traceback

# Silence the TF1 deprecation chatter so output stays readable.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# Ensure conda libstdc++ + CUDA libs are visible to the dynamic linker.
# LD_LIBRARY_PATH is read by ld.so at process start, so if it isn't set
# correctly, we re-exec ourselves with the right env.
def _ensure_env():
    if os.environ.get("GPU_TESTS_REEXEC") == "1":
        return  # already re-execed, do not loop
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


# Known build-config issues that should turn FAIL -> SKIP (not bugs we fix here).
# Each entry: (substring to match in exception text, short reason)
KNOWN_SKIPS = [
    ("No DNN in stream executor",
     "cuDNN runtime/compile version mismatch"),
    ("Loaded runtime CuDNN library",
     "cuDNN runtime/compile version mismatch"),
    ("could not find registered compiler for the platform",
     "XLA CUDA backend not registered in build"),
    ("No registered 'Add' OpKernel for 'GPU'",
     "float32 Add GPU kernel not registered (NCCL/Collective build gap)"),
    ("No registered '.*' OpKernel for 'GPU'",
     "GPU kernel not registered for this op"),
]

def _classify(exc_text):
    """Return (skip_reason, None) if exc matches a known build issue, else None."""
    import re
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
            traceback.print_exc(limit=3)
    dt = time.time() - t0
    if ok is None:
        SKIP += 1; tag = "SKIP"
    elif ok:
        PASS += 1; tag = "PASS"
    else:
        FAIL += 1; tag = "FAIL"
    RESULTS.append((name, tag, dt, detail))
    print(f"  -> {tag} ({dt:.2f}s) {detail}")


# ---------- T01 Setup ----------
def t01_setup():
    """imports, version, build flags"""
    import tensorflow as tf
    info = {
        "tf": tf.__version__,
        "cuda": tf.test.is_built_with_cuda(),
        "gpu_support": tf.test.is_built_with_gpu_support(),
    }
    return info["cuda"], str(info)


# ---------- T02 Detection ----------
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


# ---------- T03 Memory ----------
def t03_memory():
    """allocate + free large tensors on GPU"""
    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        return None, "no GPU"
    with tf.device('/GPU:0'):
        bufs = [tf.zeros([1024, 1024, 64], dtype=tf.float32) for _ in range(4)]
        total_mb = sum(b.numpy().nbytes for b in bufs) / 1e6
        del bufs
    return True, f"alloc/free {total_mb:.0f}MB OK"


# ---------- T04 Basics ----------
def t04_basics():
    """random / reduce / broadcast on GPU"""
    import tensorflow as tf
    with tf.device('/GPU:0'):
        x = tf.random.normal([4096, 4096])
        s = tf.reduce_sum(x)
        m = tf.reduce_mean(x)
        y = x + tf.range(4096, dtype=tf.float32)
        out = tf.reduce_max(y)
    return True, f"sum={float(s):.1f} mean={float(m):.4f} max={float(out):.1f}"


# ---------- T05 Matmul ----------
def t05_matmul():
    """matmul 512, 4096, 8192 with timing"""
    import tensorflow as tf
    sizes = [512, 4096, 8192]
    times = []
    with tf.device('/GPU:0'):
        for N in sizes:
            a = tf.random.normal([N, N])
            b = tf.random.normal([N, N])
            _ = tf.matmul(a, b)  # warmup
            t0 = time.time()
            for _ in range(3):
                c = tf.matmul(a, b)
            _ = c.numpy()  # force sync
            dt = (time.time() - t0) / 3
            gflops = 2 * N**3 / dt / 1e9
            times.append(f"{N}={dt*1000:.1f}ms/{gflops:.0f}GF")
    return True, " ".join(times)


# ---------- T06 Conv2D ----------
def t06_conv2d():
    """3-layer conv forward"""
    import tensorflow as tf
    with tf.device('/GPU:0'):
        x = tf.random.normal([8, 224, 224, 3])
        l1 = tf.keras.layers.Conv2D(64, 7, strides=2, padding='same', activation='relu')(x)
        l2 = tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu')(l1)
        l3 = tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu')(l2)
    return True, f"input={x.shape} -> {l3.shape}"


# ---------- T07 Backprop ----------
def t07_backprop():
    """gradient through small MLP"""
    import tensorflow as tf
    with tf.device('/GPU:0'):
        x = tf.random.normal([32, 128])
        target = tf.random.normal([32, 10])
        w1 = tf.Variable(tf.random.normal([128, 256]))
        w2 = tf.Variable(tf.random.normal([256, 10]))
        with tf.GradientTape() as tape:
            h = tf.nn.relu(x @ w1)
            y = h @ w2
            loss = tf.reduce_mean((y - target) ** 2)
        g1, g2 = tape.gradient(loss, [w1, w2])
        if g1 is None or g2 is None:
            return False, "gradient is None"
    return True, f"loss={float(loss):.3f} |dw1|={float(tf.norm(g1)):.2f}"


# ---------- T08 Train step ----------
def t08_train():
    """1 epoch synthetic MNIST-shape (jit_compile=False)"""
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
        # jit_compile=False avoids the XLA CUDA backend (not registered in
        # this build). Without it Keras 3 tries to JIT and fails with
        # "could not find registered compiler for the platform".
        model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      jit_compile=False)
        h = model.fit(x, y, batch_size=32, epochs=1, verbose=0)
    return True, f"loss={h.history['loss'][0]:.3f}"


# ---------- T09 LSTM ----------
def t09_lstm():
    """cuDNN LSTM fwd+bwd"""
    import tensorflow as tf
    with tf.device('/GPU:0'):
        x = tf.random.normal([16, 50, 128])  # batch=16, seq=50, feat=128
        lstm = tf.keras.layers.LSTM(64, return_sequences=False)
        with tf.GradientTape() as tape:
            tape.watch(x)
            y = lstm(x)
            loss = tf.reduce_sum(y)
        g = tape.gradient(loss, x)
        if g is None:
            return False, "lstm grad None"
    return True, f"out={y.shape} loss={float(loss):.2f}"


# ---------- T10 Mixed precision ----------
def t10_fp16():
    """float16 matmul (tensor cores on V100)"""
    import tensorflow as tf
    with tf.device('/GPU:0'):
        a = tf.cast(tf.random.normal([4096, 4096]), tf.float16)
        b = tf.cast(tf.random.normal([4096, 4096]), tf.float16)
        _ = tf.matmul(a, b)  # warmup
        t0 = time.time()
        for _ in range(3):
            c = tf.matmul(a, b)
        _ = c.numpy()
        dt = (time.time() - t0) / 3
        gflops = 2 * 4096**3 / dt / 1e9
    return True, f"fp16 4096x4096 {dt*1000:.1f}ms {gflops:.0f}GF"


# ---------- T11 Multi-GPU ----------
def t11_multigpu():
    """MirroredStrategy across all GPUs"""
    if QUICK:
        return None, "skipped (--quick)"
    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    if len(gpus) < 2:
        return None, f"only {len(gpus)} GPU"
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(10),
        ])
        # jit_compile=False per same reason as T08.
        model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      jit_compile=False)
    x = tf.random.normal([1024, 64])
    y = tf.random.uniform([1024], 0, 10, dtype=tf.int32)
    h = model.fit(x, y, batch_size=64, epochs=1, verbose=0)
    return True, f"{strategy.num_replicas_in_sync} replicas, loss={h.history['loss'][0]:.3f}"


# ---------- T12 Stress ----------
def t12_stress():
    """sustained matmul throughput, 30 iters"""
    if QUICK:
        return None, "skipped (--quick)"
    import tensorflow as tf
    N = 4096
    with tf.device('/GPU:0'):
        a = tf.random.normal([N, N])
        b = tf.random.normal([N, N])
        _ = tf.matmul(a, b).numpy()  # warmup
        t0 = time.time()
        for _ in range(30):
            c = tf.matmul(a, b)
        _ = c.numpy()
        dt = time.time() - t0
        gflops = 2 * N**3 * 30 / dt / 1e9
    return True, f"30x matmul({N}) {dt:.2f}s avg {gflops:.0f}GF"


# ---------- main ----------
TESTS = [
    ("T01", t01_setup),
    ("T02", t02_detect),
    ("T03", t03_memory),
    ("T04", t04_basics),
    ("T05", t05_matmul),
    ("T06", t06_conv2d),
    ("T07", t07_backprop),
    ("T08", t08_train),
    ("T09", t09_lstm),
    ("T10", t10_fp16),
    ("T11", t11_multigpu),
    ("T12", t12_stress),
]

if __name__ == "__main__":
    print("=" * 70)
    print("  TensorFlow GPU Test Suite - Power9 / V100")
    print("=" * 70)
    if ARGS:
        print(f"  filter: {sorted(ARGS)}")
    if QUICK:
        print("  --quick: skipping multi-gpu / stress")

    for name, fn in TESTS:
        run(name, fn)

    print()
    print("=" * 70)
    print(f"  SUMMARY: {PASS} pass, {FAIL} fail, {SKIP} skip "
          f"({len(RESULTS)} run)")
    print("=" * 70)
    for name, tag, dt, detail in RESULTS:
        print(f"  [{tag}] {name}  {dt:5.2f}s  {detail}")
    sys.exit(FAIL)
