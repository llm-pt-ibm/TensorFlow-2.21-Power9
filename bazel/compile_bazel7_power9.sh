#!/bin/bash
set -e
set -o pipefail

echo "=========================================================="
echo " STARTING THE NIGHTMARE: Compiling BAZEL 7.1.0 on Power9"
echo "=========================================================="

echo "1. Installing heavy dependencies (Java 21 and GCC)..."
sudo dnf install -y gcc gcc-c++ java-21-openjdk-devel zip unzip python3 wget

echo "2. Creating working directory..."
mkdir -p ~/bazel_source
cd ~/bazel_source

echo "3. Downloading Bazel 7.1.0 distribution source code..."
if [ ! -f "bazel-7.1.0-dist.zip" ]; then
    wget -q --show-progress https://github.com/bazelbuild/bazel/releases/download/7.1.0/bazel-7.1.0-dist.zip
fi

echo "4. Unpacking (this might take a while)..."
rm -rf bazel-7.1.0
mkdir bazel-7.1.0
cd bazel-7.1.0
unzip -q ../bazel-7.1.0-dist.zip

echo "5. Configuring environment for Bootstrap..."
# Bazel needs to point to local Java and limit RAM to avoid killing the VM
export EXTRA_BAZEL_ARGS="--host_javabase=@local_jdk//:jdk --local_ram_resources=HOST_RAM*.6"
export JAVA_HOME=/usr/lib/jvm/java-21-openjdk

echo "6. Starting Bazel compilation (MAY TAKE OVER 1 HOUR)..."
env EXTRA_BAZEL_ARGS="${EXTRA_BAZEL_ARGS}" bash ./compile.sh

echo "7. Installing the generated binary..."
sudo cp output/bazel /usr/local/bin/bazel
sudo chmod +x /usr/local/bin/bazel

echo "=========================================================="
echo " SUCCESS! BAZEL 7 IS READY!"
echo " Installed version:"
bazel --version
echo " Now you can go back to the TensorFlow 2.21 tutorial!"
echo " Remember to SKIP the part about installing Bazel via Conda."
echo "=========================================================="
