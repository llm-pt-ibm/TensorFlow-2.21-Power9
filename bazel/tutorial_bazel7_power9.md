# Step-by-Step: Compiling Bazel 7.1.0 on Power9 (ppc64le)

Execute the blocks below one by one in your Power9 VM terminal.
**Warning:** This is a resource-intensive process (bootstrapping). It may take 1 to 2 hours depending on your VM's cores.

### Step 1: Install Heavy System Dependencies
Bazel 7 strictly requires Java 21 to compile from scratch.

```bash
sudo dnf install -y gcc gcc-c++ java-21-openjdk-devel zip unzip python3 wget
```

### Step 2: Create and Enter a Working Directory
```bash
mkdir -p ~/bazel_source
cd ~/bazel_source
```

### Step 3: Download the Distribution Source Code
**Warning:** It must be the `-dist.zip` file (it contains necessary generated files for bootstrapping that are not included in a git clone).

```bash
wget -q --show-progress https://github.com/bazelbuild/bazel/releases/download/7.1.0/bazel-7.1.0-dist.zip
```

### Step 4: Unpack the Code
```bash
rm -rf bazel-7.1.0
mkdir bazel-7.1.0
cd bazel-7.1.0
unzip -q ../bazel-7.1.0-dist.zip
```

### Step 5: Export Compilation Variables
This is the most important step. Bazel needs to know which Java to use internally, and we need to limit the RAM.

```bash
export EXTRA_BAZEL_ARGS="--host_javabase=@local_jdk//:jdk --local_ram_resources=HOST_RAM*.6"
export JAVA_HOME=/usr/lib/jvm/java-21-openjdk
```

### Step 6: Start Compilation (The Bootstrap)
Let it run and go grab a coffee (or watch a movie).

```bash
env EXTRA_BAZEL_ARGS="${EXTRA_BAZEL_ARGS}" bash ./compile.sh
```

### Step 7: Install the Generated Binary
After successful completion, the executable file will be in the `output` folder. Let's move it to the main Linux binary path.

```bash
sudo cp output/bazel /usr/local/bin/bazel
sudo chmod +x /usr/local/bin/bazel
```

### Step 8: Validate the Installation
```bash
bazel --version
# Output should show: bazel 7.1.0
```

---

After successfully completing this, you will have native Bazel 7 on your machine. 
With this, you can return to the `tutorial_tf221_power9.md` file, completely skip Step 2 and Step 4, and go straight to the TensorFlow 2.21 build!
