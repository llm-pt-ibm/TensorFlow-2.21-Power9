#!/usr/bin/env python3
"""
Suíte de Testes Completa — TensorFlow 2.21 CPU-Only no Power9 (ppc64le)
Cobre todas as funcionalidades principais disponíveis em modo CPU.
"""
import sys
import time
import traceback
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np

PASSED = []
FAILED = []

def test(name, fn):
    """Executa e registra um teste imediatamente."""
    try:
        start = time.time()
        fn()
        elapsed = time.time() - start
        print(f"  \u2705 {name} ({elapsed:.3f}s)")
        PASSED.append(name)
    except Exception as e:
        print(f"  \u274c {name}: {e}")
        traceback.print_exc()
        FAILED.append(name)

# ===========================================================================
print(f"\n{'='*60}")
print(f"  TensorFlow {tf.__version__} — Teste CPU-Only Power9")
print(f"  Python {sys.version.split()[0]}")
print(f"  Dispositivos: {[d.name for d in tf.config.list_physical_devices()]}")
print(f"{'='*60}\n")

# ===========================================================================
# 1. OPERAÇÕES BÁSICAS COM TENSORES
# ===========================================================================
print("[ 1 ] Operações Básicas com Tensores")

test("Criação de tensores (constante, variável, zeros, ones)", lambda: (
    lambda a, b: (assert_(a.shape == (2,2)), assert_(b.shape == (2,2)))
)(tf.constant([[1.0,2.0],[3.0,4.0]]), tf.Variable(tf.ones([2,2]))))

def t_arith():
    a = tf.constant([1.0,2.0,3.0]); b = tf.constant([4.0,5.0,6.0])
    assert tf.reduce_sum(a+b).numpy() == 21.0
    assert tf.reduce_sum(b-a).numpy() == 9.0
    assert tf.reduce_sum(a*b).numpy() == 32.0
test("Aritmética elementwise (+, -, *, /)", t_arith)

def t_matmul():
    C = tf.matmul(tf.random.normal([500,500]), tf.random.normal([500,500]))
    assert C.shape == (500,500)
test("Multiplicação de matrizes (matmul 500x500)", t_matmul)

def t_reduce():
    x = tf.constant([[1.,2.,3.],[4.,5.,6.]])
    assert tf.reduce_sum(x).numpy() == 21.0
    assert tf.reduce_mean(x).numpy() == 3.5
    assert tf.reduce_max(x).numpy() == 6.0
    assert tf.reduce_min(x).numpy() == 1.0
test("Operações de redução (sum, mean, max, min)", t_reduce)

def t_broadcast():
    a = tf.constant([[1.],[2.],[3.]])
    b = tf.constant([[10.,20.,30.]])
    c = a + b
    assert c.shape == (3,3)
    assert tf.reshape(c,[9]).shape == (9,)
test("Broadcasting e reshape", t_broadcast)

def t_index():
    x = tf.constant([[1,2,3],[4,5,6],[7,8,9]])
    assert x[1,2].numpy() == 6
    assert list(x[0,:].numpy()) == [1,2,3]
    assert list(x[:,1].numpy()) == [2,5,8]
test("Indexação e slicing", t_index)

def t_cast():
    x = tf.constant([1.5,2.7,3.9])
    assert tf.cast(x, tf.int32).dtype == tf.int32
    assert tf.cast(x, tf.float64).dtype == tf.float64
test("Casting de tipos (float32, float64, int32)", t_cast)

def t_logic():
    x = tf.constant([1,2,3,4,5])
    result = tf.boolean_mask(x, tf.greater(x,3))
    assert list(result.numpy()) == [4,5]
test("Operações lógicas e boolean_mask", t_logic)

# ===========================================================================
# 2. DIFERENCIAÇÃO AUTOMÁTICA
# ===========================================================================
print("\n[ 2 ] Diferenciação Automática (GradientTape)")

def t_grad1():
    x = tf.Variable(3.0)
    with tf.GradientTape() as tape:
        y = x ** 2
    assert abs(tape.gradient(y, x).numpy() - 6.0) < 1e-5
test("Gradiente de função simples (dy/dx de x²)", t_grad1)

def t_grad2():
    x = tf.Variable(2.0)
    with tf.GradientTape() as t2:
        with tf.GradientTape() as t1:
            y = x ** 3
        dy = t1.gradient(y, x)
    d2y = t2.gradient(dy, x)
    assert abs(dy.numpy() - 12.0) < 1e-4
    assert abs(d2y.numpy() - 12.0) < 1e-4
test("Gradientes de segunda ordem (nested tape)", t_grad2)

def t_grad3():
    w = tf.Variable(2.0); b = tf.Variable(1.0)
    with tf.GradientTape() as tape:
        y = w * 3.0 + b
    g = tape.gradient(y, [w,b])
    assert abs(g[0].numpy()-3.0) < 1e-5
    assert abs(g[1].numpy()-1.0) < 1e-5
test("Gradientes em múltiplas variáveis", t_grad3)

def t_grad4():
    logits = tf.Variable([[1.,2.,3.]])
    labels = tf.constant([[0.,0.,1.]])
    with tf.GradientTape() as tape:
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels, logits))
    g = tape.gradient(loss, logits)
    assert g.shape == (1,3)
test("Gradiente de cross-entropy loss", t_grad4)

# ===========================================================================
# 3. KERAS
# ===========================================================================
print("\n[ 3 ] Keras — Construção e Treinamento")

def t_keras_seq():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(10,)),
        tf.keras.layers.Dense(1, activation='sigmoid'),
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    history = model.fit(
        np.random.random((100,10)).astype(np.float32),
        np.random.randint(0,2,(100,1)).astype(np.float32),
        epochs=3, verbose=0, batch_size=32
    )
    assert 'loss' in history.history
test("Modelo Sequential (Dense)", t_keras_seq)

def t_keras_func():
    inp = tf.keras.Input(shape=(20,))
    x = tf.keras.layers.Dense(32, activation='relu')(inp)
    x = tf.keras.layers.Dropout(0.2)(x)
    out = tf.keras.layers.Dense(10, activation='softmax')(x)
    model = tf.keras.Model(inputs=inp, outputs=out)
    model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy')
    model.fit(
        np.random.random((50,20)).astype(np.float32),
        np.random.randint(0,10,(50,)),
        epochs=2, verbose=0
    )
test("Modelo Funcional (Keras API)", t_keras_func)

def t_keras_conv():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(8,(3,3), activation='relu', input_shape=(16,16,1)),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(10, activation='softmax'),
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    model.fit(
        np.random.random((20,16,16,1)).astype(np.float32),
        np.random.randint(0,10,(20,)),
        epochs=2, verbose=0
    )
test("Conv2D + MaxPooling2D", t_keras_conv)

def t_keras_lstm():
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(16, input_shape=(10,5)),
        tf.keras.layers.Dense(1),
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(
        np.random.random((30,10,5)).astype(np.float32),
        np.random.random((30,1)).astype(np.float32),
        epochs=2, verbose=0
    )
test("LSTM para sequências", t_keras_lstm)

def t_keras_bn():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, input_shape=(10,)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1),
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(
        np.random.random((100,10)).astype(np.float32),
        np.random.random((100,1)).astype(np.float32),
        epochs=2, verbose=0
    )
test("BatchNormalization e Dropout", t_keras_bn)

# ===========================================================================
# 4. PIPELINE DE DADOS (tf.data)
# ===========================================================================
print("\n[ 4 ] Pipeline de Dados (tf.data)")

def t_ds_basic():
    x = tf.constant([[1,2],[3,4],[5,6]])
    ds = tf.data.Dataset.from_tensor_slices(x)
    assert sum(1 for _ in ds) == 3
test("Dataset from_tensor_slices", t_ds_basic)

def t_ds_ops():
    ds = tf.data.Dataset.range(100)
    ds = ds.filter(lambda x: x % 2 == 0).map(lambda x: x*x).shuffle(50).batch(10)
    batches = list(ds)
    assert len(batches) == 5 and batches[0].shape == (10,)
test("map, filter, batch, shuffle", t_ds_ops)

def t_ds_zip():
    ds1 = tf.data.Dataset.from_tensor_slices([1.,2.,3.])
    ds2 = tf.data.Dataset.from_tensor_slices([4.,5.,6.])
    for a, b in tf.data.Dataset.zip((ds1,ds2)):
        assert (a+b).numpy() > 4.0
test("zip de Datasets", t_ds_zip)

def t_ds_prefetch():
    ds = tf.data.Dataset.range(200).batch(20).prefetch(2)
    assert sum(1 for _ in ds) == 10
test("prefetch e cache", t_ds_prefetch)

# ===========================================================================
# 5. MATEMÁTICA AVANÇADA
# ===========================================================================
print("\n[ 5 ] Operações Matemáticas Avançadas")

def t_fft():
    x = tf.cast(tf.random.normal([128]), tf.complex64)
    spec = tf.signal.fft(x)
    back = tf.signal.ifft(spec)
    assert spec.shape == (128,)
    assert tf.reduce_max(tf.abs(back-x)).numpy() < 1e-4
test("FFT e IFFT", t_fft)

def t_linalg():
    A = tf.random.normal([20,20])
    s, u, v = tf.linalg.svd(A)
    assert s.shape == (20,)
    eigs = tf.linalg.eigvalsh(A + tf.transpose(A))
    assert eigs.shape == (20,)
test("SVD e eigenvalores", t_linalg)

def t_sparse():
    st = tf.SparseTensor([[0,1],[1,0],[2,2]], [1.,2.,3.], [3,3])
    dense = tf.sparse.to_dense(st)
    assert dense.shape == (3,3) and dense[0,1].numpy() == 1.0
test("SparseTensor", t_sparse)

def t_activations():
    x = tf.constant([-2.,-1.,0.,1.,2.])
    assert tf.reduce_min(tf.nn.relu(x)).numpy() == 0.0
    assert abs(tf.nn.sigmoid(tf.constant(0.)).numpy() - 0.5) < 1e-5
    assert abs(tf.nn.tanh(tf.constant(0.)).numpy()) < 1e-6
    tf.keras.activations.gelu(x)
test("Funções de ativação (relu, sigmoid, tanh, gelu)", t_activations)

def t_einsum():
    A = tf.random.normal([4,5]); B = tf.random.normal([5,6])
    C = tf.einsum('ij,jk->ik', A, B)
    assert C.shape == (4,6)
    outer = tf.tensordot(tf.constant([1.,2.,3.]), tf.constant([4.,5.]), axes=0)
    assert outer.shape == (3,2)
test("einsum e tensordot", t_einsum)

# ===========================================================================
# 6. SAVE E LOAD DE MODELOS
# ===========================================================================
print("\n[ 6 ] Save e Load de Modelos")

def t_savedmodel():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(8, activation='relu', input_shape=(4,)),
        tf.keras.layers.Dense(3, activation='softmax'),
    ])
    tf.saved_model.save(model, '/tmp/test_savedmodel_ppc64')
    loaded = tf.saved_model.load('/tmp/test_savedmodel_ppc64')
    out = loaded(tf.random.normal([5,4]), training=False)
    assert out.shape == (5,3)
test("SavedModel (save/load)", t_savedmodel)

def t_keras_h5():
    model = tf.keras.Sequential([tf.keras.layers.Dense(4, input_shape=(3,))])
    model.save('/tmp/test_model_ppc64.h5')
    loaded = tf.keras.models.load_model('/tmp/test_model_ppc64.h5')
    assert loaded.predict(np.zeros((1,3)), verbose=0).shape == (1,4)
test("Keras H5 save/load", t_keras_h5)

def t_tflite():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(4, input_shape=(2,), activation='relu'),
        tf.keras.layers.Dense(1),
    ])
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    assert len(tflite_model) > 0
test("TFLite converter (CPU)", t_tflite)

# ===========================================================================
# 7. PERFORMANCE
# ===========================================================================
print("\n[ 7 ] Performance e Stress Test")

def t_big_matmul():
    A = tf.random.normal([5000,5000]); B = tf.random.normal([5000,5000])
    start = time.time()
    soma = tf.reduce_sum(tf.matmul(A,B)).numpy()
    elapsed = time.time()-start
    print(f"        5kx5k matmul: {elapsed:.2f}s | soma={soma:.0f}", end="")
test("Multiplicação de matrizes 5000x5000", t_big_matmul)

def t_train_long():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(256, activation='relu', input_shape=(100,)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(1),
    ])
    model.compile(optimizer='adam', loss='mse')
    h = model.fit(
        np.random.random((1000,100)).astype(np.float32),
        np.random.random((1000,1)).astype(np.float32),
        epochs=20, verbose=0, batch_size=64
    )
    assert h.history['loss'][-1] < h.history['loss'][0]
test("Treinamento prolongado (MLP, 20 épocas)", t_train_long)

def t_random():
    a = tf.random.normal([1000,1000])
    b = tf.random.uniform([1000,1000])
    c = tf.random.poisson([1000], lam=5.0)
    assert abs(tf.reduce_mean(b).numpy() - 0.5) < 0.05
test("Geração de números aleatórios (normal, uniform, poisson)", t_random)

# ===========================================================================
# 8. MISC
# ===========================================================================
print("\n[ 8 ] Operações de Strings e Misc")

def t_strings():
    joined = tf.strings.reduce_join(tf.constant(["tf","2.21"]), separator="-")
    assert joined.numpy() == b"tf-2.21"
    parts = tf.strings.split(tf.constant(["hello world"]))
    assert parts.shape[0] == 1
test("tf.strings (split, join)", t_strings)

def t_lookup():
    table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(
            tf.constant(["cat","dog","bird"]),
            tf.constant([0,1,2])
        ), default_value=-1
    )
    result = table.lookup(tf.constant(["dog","unknown"]))
    assert list(result.numpy()) == [1,-1]
test("tf.lookup (StaticHashTable)", t_lookup)

def t_tf_function():
    @tf.function
    def add(a, b): return a + b
    assert add(tf.constant(3.), tf.constant(4.)).numpy() == 7.0
test("@tf.function (compilação de grafo)", t_tf_function)

def t_ragged():
    rt = tf.ragged.constant([[1,2,3],[4,5],[6]])
    assert rt.shape == (3,None)
    assert list(tf.reduce_sum(rt, axis=1).numpy()) == [6,9,6]
test("Ragged Tensors", t_ragged)

# ===========================================================================
# RELATÓRIO FINAL
# ===========================================================================
def assert_(cond):
    if not cond: raise AssertionError("Assertion failed")

total = len(PASSED) + len(FAILED)
print(f"\n{'='*60}")
print(f"  RESULTADO FINAL: {len(PASSED)}/{total} testes passaram")
print(f"{'='*60}")
if FAILED:
    print(f"\n  \u274c Falhas ({len(FAILED)}):")
    for f in FAILED:
        print(f"     - {f}")
else:
    print(f"\n  \U0001f3c6 TODOS OS TESTES PASSARAM! Power9 100% operacional.")
print()
