# Passo a Passo: Compilando Bazel 7.1.0 no Power9 (ppc64le)

Execute os blocos abaixo um por vez no terminal da sua VM Power9.
**Atenção:** Esse é um processo pesado (bootstrapping). Pode demorar de 1 a 2 horas dependendo dos núcleos da sua VM.

### Passo 1: Instalar dependências de sistema pesadas
O Bazel 7 exige obrigatoriamente o Java 21 para compilar do zero.

```bash
sudo dnf install -y gcc gcc-c++ java-21-openjdk-devel zip unzip python3 wget
```

### Passo 2: Criar e entrar em uma pasta de trabalho
```bash
mkdir -p ~/bazel_source
cd ~/bazel_source
```

### Passo 3: Baixar o código-fonte de distribuição
Atenção: tem que ser o arquivo `-dist.zip` (ele contém arquivos gerados necessários para o bootstrap que não vêm no git clone).

```bash
wget -q --show-progress https://github.com/bazelbuild/bazel/releases/download/7.1.0/bazel-7.1.0-dist.zip
```

### Passo 4: Descompactar o código
```bash
rm -rf bazel-7.1.0
mkdir bazel-7.1.0
cd bazel-7.1.0
unzip -q ../bazel-7.1.0-dist.zip
```

### Passo 5: Exportar variáveis de compilação
Este é o passo mais importante. O Bazel precisa saber qual Java usar internamente, e precisamos limitar a RAM.

```bash
export EXTRA_BAZEL_ARGS="--host_javabase=@local_jdk//:jdk --local_ram_resources=HOST_RAM*.6"
export JAVA_HOME=/usr/lib/jvm/java-21-openjdk
```

### Passo 6: Iniciar o compilação (O Bootstrap)
Deixe rodando e vá tomar um café (ou assistir a um filme).

```bash
env EXTRA_BAZEL_ARGS="${EXTRA_BAZEL_ARGS}" bash ./compile.sh
```

### Passo 7: Instalar o binário gerado
Após a conclusão bem sucedida, o arquivo executável estará na pasta `output`. Vamos movê-lo para a pasta principal do Linux.

```bash
sudo cp output/bazel /usr/local/bin/bazel
sudo chmod +x /usr/local/bin/bazel
```

### Passo 8: Validar a instalação
```bash
bazel --version
# A saída deve mostrar: bazel 7.1.0
```

---

Após concluir com sucesso, você terá o Bazel 7 nativo da sua máquina. 
Com isso, você pode voltar para o arquivo `tutorial_tf221_power9.md`, pular completamente o Passo 2 e o Passo 4, e ir direto pro abraço no TensorFlow 2.21!
