import tensorflow as tf
import time
import os
import sys

# Garante que os logs não poluam o teste visualmente, a menos que ocorra erro
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def print_header(title):
    print(f"\n" + "="*60)
    print(f" 🚀 {title}")
    print("="*60)

def run_test(name, func):
    print(f"⏳ Testando: {name}... ", end="")
    sys.stdout.flush()
    try:
        start = time.time()
        res = func()
        end = time.time()
        print(f"✅ OK! ({end - start:.4f}s)")
        return res
    except Exception as e:
        print(f"❌ FALHOU!\n   Erro: {e}")
        return None

# ==========================================
# SETUP DE DADOS NA CPU (Foge dos stubs)
# ==========================================
print_header("Preparando Dados na CPU")
with tf.device('/CPU:0'):
    # Matrizes para cuBLAS / cuSOLVER
    mat_a = tf.random.normal([2000, 2000])
    mat_b = tf.random.normal([2000, 2000])
    
    # Tensores 4D para cuDNN (Batch, Height, Width, Channels)
    # Formato NHWC padrão
    img_batch = tf.random.normal([32, 128, 128, 3])
    conv_filter = tf.random.normal([3, 3, 3, 16])  # 16 filtros 3x3

    # Tensores para cuFFT (Transformada de Fourier)
    complex_tensor = tf.cast(tf.random.normal([1024, 1024]), tf.complex64)

# ==========================================
# TESTES NA GPU
# ==========================================
print_header("Iniciando Bateria de Testes na GPU")

with tf.device('/GPU:0'):

    # 1. cuBLAS (Operações de Álgebra Linear Básica)
    run_test(
        "Multiplicação de Matrizes Pura (cuBLAS)",
        lambda: tf.matmul(mat_a, mat_b)
    )

    # 2. cuDNN (Operações Espaciais e Convolucionais de Deep Learning)
    run_test(
        "Convolução 2D (cuDNN)",
        lambda: tf.nn.conv2d(img_batch, conv_filter, strides=[1, 1, 1, 1], padding='SAME')
    )

    run_test(
        "Max Pooling 2D (cuDNN)",
        lambda: tf.nn.max_pool2d(img_batch, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    )

    # 3. cuSOLVER / cuBLAS Avançado
    # Fatoração de Cholesky (Mede a sanidade do solver linear da GPU)
    # Criamos uma matriz simétrica definida positiva garantida na CPU
    with tf.device('/CPU:0'):
        mat_pos_def = tf.matmul(mat_a, tf.transpose(mat_a)) + tf.eye(2000) * 1e-3
    
    run_test(
        "Decomposição Cholesky (cuSOLVER)",
        lambda: tf.linalg.cholesky(mat_pos_def)
    )

    # 4. cuFFT (Transformada Rápida de Fourier)
    run_test(
        "Transformada de Fourier 2D (cuFFT)",
        lambda: tf.signal.fft2d(complex_tensor)
    )

    # 5. Alto Nível (Keras com cuDNN/cuBLAS combinados)
    def keras_forward_pass():
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(128, 128, 3)),
            tf.keras.layers.MaxPooling2D(2,2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(10)
        ])
        return model(img_batch)

    run_test(
        "Forward Pass Completo Keras CNN",
        keras_forward_pass
    )

print_header("Fim do Diagnóstico!")
print("Se a maioria dos testes acima deu 'OK', as bibliotecas dinâmicas da NVIDIA (cuBLAS, cuDNN, cuSOLVER, cuFFT) estão perfeitamente ligadas ao TensorFlow no Power9.\n")
