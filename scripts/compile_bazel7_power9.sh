#!/bin/bash
set -e
set -o pipefail

echo "=========================================================="
echo " INICIANDO O PESADELO: Compilando BAZEL 7.1.0 no Power9"
echo "=========================================================="

echo "1. Instalando dependências pesadas (Java 21 e GCC)..."
sudo dnf install -y gcc gcc-c++ java-21-openjdk-devel zip unzip python3 wget

echo "2. Criando diretório de trabalho..."
mkdir -p ~/bazel_source
cd ~/bazel_source

echo "3. Baixando o código fonte de distribuição do Bazel 7.1.0..."
if [ ! -f "bazel-7.1.0-dist.zip" ]; then
    wget -q --show-progress https://github.com/bazelbuild/bazel/releases/download/7.1.0/bazel-7.1.0-dist.zip
fi

echo "4. Descompactando (isso pode demorar um pouco)..."
rm -rf bazel-7.1.0
mkdir bazel-7.1.0
cd bazel-7.1.0
unzip -q ../bazel-7.1.0-dist.zip

echo "5. Configurando ambiente para o Bootstrap..."
# O Bazel precisa focar no Java local e limitar a RAM para não matar a VM
export EXTRA_BAZEL_ARGS="--host_javabase=@local_jdk//:jdk --local_ram_resources=HOST_RAM*.6"
export JAVA_HOME=/usr/lib/jvm/java-21-openjdk

echo "6. Iniciando a compilação do Bazel (PODE LEVAR MAIS DE 1 HORA)..."
env EXTRA_BAZEL_ARGS="${EXTRA_BAZEL_ARGS}" bash ./compile.sh

echo "7. Instalando o binário gerado..."
sudo cp output/bazel /usr/local/bin/bazel
sudo chmod +x /usr/local/bin/bazel

echo "=========================================================="
echo " SUCESSO! O BAZEL 7 FICOU PRONTO!"
echo " Versão instalada:"
bazel --version
echo " Agora você pode voltar para o tutorial do TensorFlow 2.21!"
echo " Lembre-se de PULAR a parte de instalar o bazel pelo conda."
echo "=========================================================="
