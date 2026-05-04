# Creating and Publishing a Conda Package for ppc64le

> Complete guide to packaging a native `.whl` into a Conda package and publishing it on Anaconda Cloud.

---

## Prerequisites

- Linux ppc64le with Conda installed
- `conda-build` installed
- Account on [anaconda.org](https://anaconda.org)
- A `.whl` file compiled for ppc64le

### Install conda-build

```bash
conda activate base
conda install conda-build anaconda-client -y
```

---

## Step 1 — Create the Recipe Structure

```bash
mkdir ~/tf_conda_recipe
cd ~/tf_conda_recipe
```

---

## Step 2 — Write the `meta.yaml`

The `meta.yaml` is the package recipe. It defines the name, version, source, installation script, and all dependencies.

```bash
cat << 'EOF' > meta.yaml
package:
  name: tensorflow-cpu
  version: "2.21.0"

source:
  url: https://github.com/llm-pt-ibm/tensorflow/releases/download/v2.21.0-cpu-only/tensorflow-2.21.0-cp311-cp311-linux_ppc64le.whl

build:
  number: 0
  script: pip install --no-deps $SRC_DIR/tensorflow-2.21.0-cp311-cp311-linux_ppc64le.whl

requirements:
  host:
    - python =3.11
    - pip
  run:
    - python >=3.11,<3.12
    - numpy >=2.0.0
    - grpcio
    - h5py <3.15.0
    - protobuf >=6.31.1,<8.0.0
    - ml_dtypes
    - libstdcxx-ng
    - absl-py
    - astunparse
    - gast
    - google-pasta
    - keras >=2.0.0
    - opt_einsum
    - python-flatbuffers
    - requests
    - six
    - termcolor
    - typing_extensions
    - wrapt
    - libclang

about:
  home: https://github.com/llm-pt-ibm/tensorflow
  license: Apache-2.0
  summary: "TensorFlow 2.21.0 (CPU Only) natively compiled for ppc64le"
EOF
```

### `meta.yaml` Structure

| Section | Description |
|---|---|
| `package` | Final package name and version |
| `source.url` | URL of the `.whl` to be downloaded and packaged |
| `build.script` | Command that installs the `.whl` in the build environment |
| `requirements.host` | Dependencies required **during the build** |
| `requirements.run` | Dependencies installed **along with the package** for the end user |
| `about` | Metadata (homepage, license, description) |

---

## Step 3 — Run the Build

```bash
conda activate base

# Clear previous build cache
conda build purge

# Build the package
conda build .
```

The process will:
1. Download the `.whl` from the declared URL
2. Create an isolated build environment
3. Install the package in that environment
4. Package everything into a `.conda` file

At the end, the terminal will display the path of the generated file:
```
/root/miniforge3/conda-bld/linux-ppc64le/tensorflow-cpu-2.21.0-py311_0.conda
```

### Locate the Generated File

```bash
find ~/miniforge3/conda-bld -name "tensorflow-cpu*.conda"
```

---

## Step 4 — Test Locally Before Uploading

```bash
# Create a clean environment simulating the user's machine
conda create -n tf_test python=3.11 -y
conda activate tf_test

# Install using the local package
conda install -c conda-forge --use-local tensorflow-cpu=2.21.0 -y

# Verify
python -c "import tensorflow as tf; print(tf.__version__)"
```

**Expected output:** `2.21.0`

---

## Step 5 — Generate Token on Anaconda

1. Access [anaconda.org/settings/access](https://anaconda.org/settings/access)
2. Click on **"Generate Token"**
3. Check permissions: **conda** and **write**
4. Copy the generated token

> ⚠️ Keep the token in a safe place — it will not be displayed again.

---

## Step 6 — Upload to Anaconda Cloud

```bash
# Fix file permissions
chmod 644 /root/miniforge3/conda-bld/linux-ppc64le/tensorflow-cpu-2.21.0-py311_0.conda

# Upload (replace YOUR_TOKEN with the generated token)
anaconda -t YOUR_TOKEN upload \
  /root/miniforge3/conda-bld/linux-ppc64le/tensorflow-cpu-2.21.0-py311_0.conda
```

**Expected output:**
```
Using "your-user" as upload username
Processing "tensorflow-cpu-2.21.0-py311_0.conda"
Creating package "tensorflow-cpu"
Creating release "2.21.0"
Uploading file "your-user/tensorflow-cpu/2.21.0/linux-ppc64le/..."
Upload complete
conda located at:
  https://anaconda.org/your-user/tensorflow-cpu
```

---

## Step 7 — Verify on the Website

Access in your browser:
```
https://anaconda.org/YOUR_USERNAME/tensorflow-cpu
```

---

## Step 8 — Test by Installing from the Remote Channel

```bash
# Completely clean environment
conda create -n tf_channel_test python=3.11 -y
conda activate tf_channel_test

# Install directly from the public channel (without --use-local)
conda install -c YOUR_USERNAME -c conda-forge tensorflow-cpu=2.21.0 -y

# Confirm package origin
conda list | grep tensorflow
```

**Expected output:**
```
tensorflow-cpu   2.21.0   py311_0   your-user
```

---

## Automatic Upload (optional)

To have every `conda build` upload automatically upon completion:

```bash
conda config --set anaconda_upload yes
```

---

## Distributing to Other ppc64le Servers

Any ppc64le server can install with a single command:

```bash
conda install -c YOUR_USERNAME -c conda-forge tensorflow-cpu=2.21.0 -y
```

Or with channels configured globally on the machine:

```bash
# One-time configuration on the machine
conda config --add channels conda-forge
conda config --add channels YOUR_USERNAME

# Simple installation from now on
conda install tensorflow-cpu=2.21.0 -y
```

---

## Summary of Commands

```bash
# 1. Create recipe
mkdir ~/tf_conda_recipe && cd ~/tf_conda_recipe
# (create meta.yaml as shown above)

# 2. Build
conda activate base
conda build purge
conda build .

# 3. Test local
conda create -n tf_test python=3.11 -y
conda activate tf_test
conda install -c conda-forge --use-local tensorflow-cpu=2.21.0 -y

# 4. Upload
chmod 644 /root/miniforge3/conda-bld/linux-ppc64le/tensorflow-cpu-2.21.0-py311_0.conda
anaconda -t YOUR_TOKEN upload /root/miniforge3/conda-bld/linux-ppc64le/tensorflow-cpu-2.21.0-py311_0.conda

# 5. Install from public channel
conda install -c YOUR_USERNAME -c conda-forge tensorflow-cpu=2.21.0 -y
```
