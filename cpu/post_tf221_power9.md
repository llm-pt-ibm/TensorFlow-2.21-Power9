# TensorFlow 2.21 no IBM Power9 (ppc64le): Da Análise à Compilação Nativa

**Autor:** Yalle Rocha Silva  
**Data:** 18 de Março de 2026  
**Plataforma de teste:** IBM Power9 — AlmaLinux — ppc64le

---

## Contexto

O TensorFlow é o framework de machine learning mais adotado globalmente, servindo como dependência para centenas de ferramentas de IA. No entanto, desde 2021, o Google encerrou o suporte oficial de binários pré-compilados para a arquitetura ppc64le, e o repositório comunitário `tensorflow/community` foi arquivado em 2025.

Historicamente, a IBM liderou o suporte ao TensorFlow no Power por meio do IBM PowerAI (2016) e do Watson Machine Learning Community Edition — WMLCE (2019). Essas iniciativas foram descontinuadas, e a responsabilidade foi transferida ao projeto comunitário Open-CE e à distribuição enterprise RocketCE da Rocket Software. Este cenário levanta questões críticas: quais ferramentas estão bloqueadas? O que é possível executar hoje? E o que pode ser feito para avançar o ecossistema?

---

## O Que Fizemos

Realizamos quatro etapas de investigação:

1. Mapeamos quais ferramentas estavam **bloqueadas** pela ausência do TensorFlow no Power9.
2. Instalamos e validamos o **TensorFlow 2.14** em uma VM real via RocketCE.
3. Analisamos o que essa versão **perde** em relação à 2.21 (a mais recente para x86).
4. **Compilamos com sucesso o TensorFlow 2.21 do zero**, diretamente no Power9, tornando-nos pioneiros nessa façanha fora dos laboratórios do Google.

---

## Ferramentas Bloqueadas Sem o TensorFlow

Sem o TensorFlow instalado, todo o ecossistema de deep learning fica inacessível no Power9. Isso inclui o próprio TensorFlow Core, o Keras (API de alto nível para redes neurais), o TensorBoard (visualização de treinos), o TF Hub (modelos pré-treinados), e o pipeline de produção TFX, que engloba validação de dados, transformação, análise de modelos e serving.

Ferramentas de nível mais alto também são afetadas: Hugging Face Transformers (no backend TF), Keras Tuner, TF-Agents para reinforcement learning, e TensorFlow Datasets.

### O Impacto Direto no Portfólio IBM

A IBM é uma das maiores consumidoras corporativas do TensorFlow, e a ausência de suporte no Power9 cria uma contradição interna crítica: o hardware mais avançado da IBM para IA não consegue executar seu próprio software de IA. As ferramentas diretamente afetadas incluem:

| Ferramenta IBM | Dependência do TensorFlow | Limitação no Power9 |
|----------------|--------------------------|---------------------|
| **Watson Studio** | Backend principal para deep learning notebooks (TF/Keras) | Deep Learning jobs declarados oficialmente como não suportados em ppc64le |
| **Watson Machine Learning (WML)** | Deploy e serving de modelos TensorFlow via API REST | Implantação de modelos TF bloqueada na plataforma Power |
| **Watson Studio AutoAI** | Pipeline automatizado usa TF por baixo para redes neurais | Funcionalidades de deep learning desabilitadas no Power |
| **IBM Cloud Pak for Data** | Integra Watson Studio + WML; deep learning depende de TF | Capacidades de IA reduzidas em instalações on-prem no Power |
| **IBM Watson Natural Language Understanding** | Utiliza otimizações TensorFlow (via Intel oneDNN) para throughput de NLP | Performance reduzida e fine-tuning de modelos customizados limitado |
| **IBM Research — projetos internos** | Diversos projetos de pesquisa dependem de TF para treinamento e reprodutibilidade | Reprodutibilidade em hardware Power comprometida |

Em resumo: o Power9 é a plataforma de IA da IBM para clientes enterprise — bancos, seguradoras, órgãos governamentais — e a ausência de um TensorFlow moderno cria um gap funcional entre o hardware vendido e o software prometido. Ter o TF 2.21 nativo para ppc64le não é apenas uma conquista técnica; é um requisito para que o Power9 cumpra seu papel como plataforma de IA de ponta a ponta.

---

## Instalação do TF 2.14 via RocketCE (Ponto de Partida)

Validamos a instalação do TensorFlow 2.14.1 em uma VM Power9 com AlmaLinux, usando Miniforge (conda). O processo é direto:

```bash
conda create -n tf214 python=3.11 -y
conda activate tf214
conda install -c rocketce tensorflow-cpu=2.14.1 -y
```

O resultado foi o TensorFlow 2.14.1 funcional, com apenas um warning menor sobre versão do NumPy (resolvível atualizando para NumPy ≥ 1.26.4). Essa mesma versão também está disponível nos canais Open-CE da Oregon State University e do MIT.

Com o TF 2.14 funcionando, desbloqueamos imediatamente: Keras (`tf.keras`), TensorBoard, TensorFlow Hub, `tensorflow-text`, Hugging Face Transformers, Jupyter, e todo o stack de ML clássico (scikit-learn, XGBoost, LightGBM, pandas). O Open-CE 1.11 também traz PyTorch, ONNX Runtime e até LLaMA.cpp para inferência de LLMs.

---

## O Que o TF 2.14 Perde em Relação ao 2.21

A versão 2.14 é funcional, mas está à distância de diversas versões. As perdas mais significativas se concentram em duas áreas:

### Keras 3 (a partir do TF 2.16)

O Keras 3 é uma reescrita completa que transforma o Keras em um framework **multi-backend** — o mesmo modelo, o mesmo código, rodando em TensorFlow, PyTorch ou JAX sem qualquer alteração. Para o Power9, isso tem implicações práticas imediatas:

- **`keras.ops` e `keras.random`**: API unificada de operações numéricas que funciona identicamente nos três backends, eliminando o `tf.*` hardcoded no código de pesquisa.
- **LoRA nativo (`model.enable_lora()` via KerasHub)**: Fine-tuning eficiente de LLMs sem bibliotecas externas — reduz significativamente a memória necessária para adaptação de modelos fundacionais.
- **Serialização unificada**: Modelos salvos com `model.save()` no Keras 3 são portaveis entre backends — um modelo treinado com backend JAX pode ser carregado e servido com backend TensorFlow.
- **Compatibilidade com PyTorch DataLoader**: O Keras 3 aceita `torch.utils.data.DataLoader` diretamente, abrindo o ecossistema de datasets do PyTorch para modelos TensorFlow.
- **Preset de modelos populares**: `keras_hub` (antigo KerasNLP + KerasCV) fornece presets prontos de BERT, GPT-2, LLaMA, CLIP, Stable Diffusion — todos acessíveis com `from_preset()` e fine-tuning imediato.

No TF 2.14, estamos presos ao Keras 2, que só funciona com TensorFlow e não recebe mais atualizações de novos recursos.

### NumPy 2.0 (a partir do TF 2.18)

Além de corrigir dezenas de inconsistências históricas da API, o NumPy 2.0 traz ganhos sólidos que repercutem diretamente no Power9:

- **Compatibilidade total do ecossistema**: SciPy ≥ 1.14, pandas ≥ 3.0, scikit-learn ≥ 1.5, matplotlib ≥ 3.9 e Hugging Face Datasets ≥ 2.20 já exigem NumPy 2.0. Sem ele, o `pip install` de qualquer dessas bibliotecas nas versões atuais gera conflito.
- **StringDType**: Novo tipo de dado `np.dtypes.StringDType` com semântica de string nativa em arrays — elimina o contorno usual com `dtype=object`, comum em pipelines de NLP.
- **Melhorias de performance em operações de cópia**: Novos protocolos `__dlpack__` e `__array__` permitem troca zero-copy de buffers entre NumPy, TensorFlow e PyTorch — especialmente relevante em pipelines de dados intensivos.
- **C API limpa e estável para extensões**: A partir do NumPy 2.0, a API C é mais consistente e favorece compatibilidade futura entre versões minor, reduzindo o ciclo de recompilação que afeta extensões como SciPy e pandas no Power9.
- **`np.exceptions` e deprecações limpas**: Remove alias de tipos legados (`np.int`, `np.float`, `np.bool`), tornando o código científico mais robusto e previsível.

Em suma: ficar no TF 2.14 não é apenas uma limitação de deep learning — é um congelamento de **todo o stack científico Python** no Power9 em versões de 2023.

Outras perdas menores incluem: suporte a Python 3.12+ (TF 2.16), o LiteRT como substituto do TFLite com até 1.4× mais performance em GPU (TF 2.19–2.21), e patches de segurança acumulados. Funcionalidades específicas de GPU (CUDA 12.3, kernels para RTX 40, TensorRT, Flash Attention) não se aplicam ao nosso cenário CPU-only.

---

## Oportunidades Inéditas

A análise revelou oportunidades que não foram exploradas por nenhuma entidade — nem IBM, nem Open-CE, nem Rocket Software:

### Portar o JAX para ppc64le

O JAX, framework de computação numérica do Google baseado em diferenciação automática e compilação JIT via XLA, simplesmente não existe para ppc64le. Se portado, desbloquearia um ecossistema inteiro: Flax para redes neurais (biblioteca oficial recomendada pelo Google DeepMind), Optax para otimizadores, NumPyro para programação probabilística, AlphaFold para biologia computacional, e T5X/MaxText para treino de LLMs. Além disso, JAX é o backend mais rápido do Keras 3 — combinado com TF 2.16+, daria ao Power9 paridade funcional completa com x86 em IA.

---

## Compilando o TensorFlow 2.21 Nativamente no Power9 (CPU-Only): Missão Cumprida

O que até aqui era visto como inviável fora dos laboratórios do Google, realizamos: **compilamos com sucesso o TensorFlow 2.21 diretamente a partir do código-fonte em uma VM Power9**, gerando um pacote `.whl` nativo para `linux_ppc64le` e validando seu funcionamento com uma suíte completa de testes. Este é um **marco fundamental de CPU-only** — a base estável sobre a qual o suporte a GPU será construído na próxima etapa.

### O Problema: Hermetismo e Dependência de x86

A arquitetura moderna do TensorFlow (e seu sistema de build, o Bazel 7) abraçou o modelo "Hermético": forçando o uso de binários pré-compilados e lógicas atreladas às arquiteturas x86_64, aarch64 e aceleradores NVIDIA. Para ppc64le, isso significa que a compilação naïve simplesmente falha ao tentar baixar ferramentas para arquiteturas incompatíveis.

Identificamos quatro categorias de bloqueio:
- **Bazel 7:** O Google não distribui o Bazel 7 para PowerPC. Seria necessário compilá-lo do zero.
- **Toolchains herméticas:** O TF 2.21 tenta baixar LLVM/Clang pré-compilado para x86 ou aarch64, que não executa no Power9.
- **Dependências CUDA/GPU:** Mesmo em modo CPU-only, o sistema de build tenta baixar e vincular bibliotecas NVIDIA gigantes. Nossa estratégia foi **isolar completamente o suporte a GPU** com stubs vazios, garantindo uma fundação CPU-only estável antes de adicionar qualquer acelerador.
- **Bugs de C++ latentes:** O código do XLA e do MLIR contém construções que funcionam no Clang do Google, mas quebram no GCC 13 padrão do sistema — de flags AVX-512 até ambiguidades de template em `absl::NoDestructor`.

### Etapa 1: Compilando o Bazel 7.1.0 do Zero

Como o Google não distribui o Bazel 7 para ppc64le, iniciamos compilando o próprio Bazel a partir do seu código-fonte de distribuição (o arquivo `-dist.zip`, que inclui os artefatos gerados necessários para o bootstrap).

O processo exige Java 21 e leva de **1 a 2 horas** dependendo dos núcleos da VM. As variáveis críticas que tornam a compilação estável são:

```bash
export EXTRA_BAZEL_ARGS="--host_javabase=@local_jdk//:jdk --local_ram_resources=HOST_RAM*.6"
export JAVA_HOME=/usr/lib/jvm/java-21-openjdk
env EXTRA_BAZEL_ARGS="${EXTRA_BAZEL_ARGS}" bash ./compile.sh
```

O resultado é o binário `output/bazel`, que instalamos em `/usr/local/bin/bazel`. Sem esse passo, nenhuma das etapas seguintes é possível.

### Etapa 2: Estratégia de Bypass — Repositórios Stub

Com o Bazel 7 funcional, atacamos o problema das dependências herméticas. Nossa solução foi criar **repositórios "stub"** — diretórios locais vazios que satisfazem as declarações de dependência do Bazel sem baixar nada:

- **LLVM stubs** (`llvm_linux_x86_64`, `llvm_linux_aarch64`, etc.): Filesgroups vazios que satisfazem as regras de toolchain sem tentar instalar o LLVM.
- **CUDA/ROCm/TensorRT stubs**: Bibliotecas C++ e regras Starlark vazias que permitem que o build prossiga em modo `NO_CUDA=1` sem erros de dependência faltante.
- **PyPI stubs**: Módulos Python stub que simulam as dependências do pip hermético do Google, forçando o uso das bibliotecas do ambiente conda.
- **Python stub**: Redireciona o `python_3_11_host` para o Python do nosso ambiente conda, contornando o download do Python hermético que não existe para ppc64le.

Todos os stubs são injetados via `--override_repository` na chamada do `bazel build`, sem alterar o código-fonte do TensorFlow.

### Etapa 3: Patches Cirúrgicos no Código-Fonte

Com a infraestrutura de build resolvida, encontramos 21 bugs de compatibilidade no código C++ e Python do TensorFlow que se manifestam exclusivamente na combinação **GCC 13 + ppc64le**. Cada um foi resolvido com um patch Python preciso:

| # | Arquivo | Problema | Solução |
|---|---------|----------|---------|
| A | `python_init_toolchains.bzl` | Tenta baixar Python hermético inexistente para ppc64le | Stubbing da função `python_init_toolchains` |
| B | `python_init_pip.bzl` | Usa `python_interpreter_target` apontando para binário hermético | Redireciona para wrapper shell do conda |
| C | `WORKSPACE` | Registra 30+ toolchains CUDA/NCCL/LLVM | Script Python comenta blocos inteiros por padrão |
| D | `builtin_fp16.h` | GCC 13 exige `#include <cstdint>` explícito | Inserção no topo do arquivo |
| E | `XLA BUILD/bzl` | Flags `-mprefer-vector-width=512` e `-fno-experimental-sanitize-metadata=all` são exclusivas do Clang | Substituídas por strings vazias via `find + sed` |
| F | `eigen_unary.cc` | `__has_builtin(__builtin_vectorelements)` quebra no GCC | Substituído por literal `0` |
| G | `boringssl/base.h` | Detecta Power9 como "Unknown target CPU" e aborta | Substituído por defines manuais de 64-bit |
| H | `xla/shape.h` e `.cc` | Disputa de `noexcept` em move constructors no GCC 13 | Remoção do `noexcept` nas declarações |
| I | `dso_loader.cc` | Inclui headers CUDA/TensorRT que não existem nos stubs | Substituído por defines manuais de versão `"0"` |
| J | `group_events.cc` | Ambiguidade de `absl::NoDestructor` com `{}` no GCC 13 | Tipo concreto explicitado: `absl::flat_hash_set<int64_t>` |
| K | `fuse_qdq_pass.cc` | Mesmo problema de `absl::NoDestructor` com `std::string` | Tipo concreto: `absl::flat_hash_set<std::string>` |
| L | `allocation_value.h` | Move constructor com `noexcept` recusado pelo GCC em vector | Reescrita da declaração sem `noexcept` |
| M | `hlo_sharding_util.h` | Ambiguidade entre `Span` e `Iota` em initializer list | Cast explícito para `absl::Span<const int64_t>` |
| N | `optimize_pass.cc` | CTAD ausente em `std::multiplies()` no GCC 8 | Tipo explicitado: `std::multiplies<int64_t>()` |
| O | `dtensor/mlir/*.cc` | `llvm::cast`, `isa`, `dyn_cast` usados sem namespace | Inserção de `using` declarations por arquivo |
| P | `xla/codegen/*.h` e `cpu/*.cc` | `noexcept = default` em múltiplos construtores | Remoção do `noexcept` via regex |
| Q | `ynn_support.cc` | `absl::NoDestructor` com tuple como tipo implícito | Tipo concreto: `absl::flat_hash_set<tuple<...>>` |
| R | `convolution_lib.h` | Captura por cópia de `absl::BlockingCounter` em lambda | Reescrita da captura com variável local |
| S | `thunk_executor.cc` | `reserve()` em vetor com tipo incompleto em GCC antigo | Remoção do `reserve()` problemático |
| T | `tensorflow.bzl` | Flag `//command_line_option:modify_execution_info` removida no Bazel 7 | Limpeza das declarações `inputs/outputs` |
| U | `build_pip_package.py` | Falhas de glob e diretórios ausentes no empacotamento | Super-patch com monkey-patching de `shutil` e `subprocess` |

Adicionalmente, o patch do `pybind11_bazel` remove a flag `-fvisibility=hidden` do macro `pybind_library`, que conflitava com nossa flag global `--copt=-fvisibility=default` necessária para que os símbolos dos módulos Python fiquem exportados corretamente.

### Etapa 4: A Compilação — ~4 Horas de Bazel

Com todos os patches aplicados, a compilação final é disparada com um único comando `bazel build`, acumulando ~80 flags de `--override_repository` para injetar todos os stubs:

```bash
bazel build \
    --config=opt \
    --define=tflite_with_xnnpack=false \
    --local_ram_resources=HOST_RAM*.6 \
    --copt=-fvisibility=default \
    --cxxopt=-fvisibility=default \
    --noincompatible_enable_cc_toolchain_resolution \
    --jobs=$(nproc) \
    # [+ ~75 flags --override_repository] \
    //tensorflow/tools/pip_package:wheel
```

O cache incremental do Bazel é fundamental aqui: cada vez que um patch é necessário e a compilação é retomada, apenas os alvos afetados são recompilados. Isso transformou o ciclo "patch → compilar → erro → patch" de inviável em gerenciável.

### Resultado: TensorFlow 2.21.0 Funcional no Power9

Após a compilação, instalamos o `.whl` gerado e executamos uma suíte completa de **35 testes**, cobrindo oito categorias funcionais:

| Categoria | Testes | Resultado |
|-----------|--------|-----------|
| Operações básicas com tensores | 7 | ✅ 7/7 |
| Diferenciação automática (GradientTape) | 4 | ✅ 4/4 |
| Keras — Sequential, Funcional, CNN, LSTM, BN | 5 | ✅ 5/5 |
| Pipeline de dados (`tf.data`) | 4 | ✅ 4/4 |
| Matemática avançada (FFT, SVD, einsum) | 5 | ✅ 5/5 |
| Save e Load de modelos (SavedModel, H5, TFLite) | 3 | ✅ 3/3 |
| Performance e stress test | 3 | ✅ 3/3 |
| Strings e misc (`tf.function`, RaggedTensor) | 4 | ✅ 4/4 |
| **TOTAL** | **35** | **✅ 35/35** |

O teste de stress (multiplicação de matrizes 5000×5000) executou com sucesso na CPU do Power9, e o treinamento de um MLP por 20 épocas confirmou convergência de loss — indicando que a diferenciação automática, otimizadores e operações numéricas estão todos funcionando corretamente de ponta a ponta.

```
============================================================
  TensorFlow 2.21.0 — Teste CPU-Only Power9
  Python 3.11.x
  Dispositivos: ['/physical_device:CPU:0']
============================================================

🏆 TODOS OS TESTES PASSARAM! Power9 100% operacional.
```

---

## Reprodutibilidade

Todo o processo está documentado em dois tutoriais completos e reproduzíveis, disponíveis neste repositório:

- **`tutorial_bazel7_power9.md`** — Compilação do Bazel 7.1.0 a partir do código-fonte (8 passos, ~1–2 horas)
- **`tutorial_tf221_power9.md`** — Compilação do TensorFlow 2.21 com todos os patches (14 passos, ~4 horas)
- **`test_tf_cpu_power9.py`** — Suíte de 35 testes para validação após a instalação

O processo foi projetado para ser **retomável**: graças ao cache do Bazel, qualquer interrupção ou novo patch não exige recompilar tudo do zero.

---

## Impacto e Próximos Passos

Esta compilação representa a **versão mais recente do TensorFlow** disponível nativamente para ppc64le fora dos laboratórios do Google. Com ela:

- ✅ **Keras 3** fica disponível para ppc64le pela primeira vez
- ✅ **NumPy 2.0** deixa de ser um gargalo para o ecossistema científico Python no Power9
- ✅ **Todo o stack Hugging Face** nas versões atuais torna-se compatível
- ✅ **LiteRT / TFLite** com performance melhorada está acessível

### Próximos Passos

**1. Suporte a GPU (CUDA) no Power9 — a fronteira imediata**

O TF 2.21 que compilamos roda exclusivamente em CPU. O próximo desafio é **repetir o processo com CUDA habilitado**, aproveitando servidores Power9 equipados com GPUs NVIDIA (como o IBM AC922, que conecta CPUs Power9 a GPUs V100 via NVLink). Os stubs que criamos para isolar a GPU nesta compilação foram projetados justamente para facilitar essa transição: ao substituí-los pelas bibliotecas CUDA reais, teremos um ponto de partida sólido para a compilação GPU. Se bem-sucedido, o Power9 passaria a ter o **framework de deep learning mais recente com aceleração de hardware**, algo inexistente hoje em qualquer distribuição para ppc64le.

**2. Portar o JAX para ppc64le**

O JAX combinado com este TensorFlow 2.21 daria ao Power9 paridade funcional completa com x86 em IA, incluindo Keras 3 com backend JAX — desbloqueando AlphaFold, T5X, MaxText e todo o ecossistema de pesquisa do Google.
