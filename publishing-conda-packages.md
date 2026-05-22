# Publicando um Pacote Binário no Canal Conda `ufcg-ibm`

Guia genérico para pegar qualquer binário pré-compilado (uma `.whl`
do Python, um tarball, um executável estático) e empacotá-lo como um
pacote Conda instalável via `-c ufcg-ibm`. Funciona para qualquer
arquitetura; os exemplos têm como alvo `linux-ppc64le`.

A experiência final do usuário é um comando só:
```bash
conda install -c ufcg-ibm -c conda-forge <pacote>=<versao> -y
```

---

## Pré-requisitos

### Máquina nativa da arquitetura alvo

`conda build` **não** faz cross-compile. Para publicar um pacote
`linux-ppc64le` você precisa rodar o build em uma máquina ppc64le real
(bare metal, VM ou container). O `.conda` gerado é específico da
plataforma e fica num subdiretório com o nome do alvo.

### Miniforge3 (ou Anaconda) com as ferramentas de build

`conda-build` é o motor que transforma a receita num artefato `.conda`;
`anaconda-client` traz o CLI `anaconda` usado no upload para a nuvem.

```bash
source ~/miniforge3/etc/profile.d/conda.sh    # necessario pra 'conda activate' funcionar em script
conda activate base
conda install conda-build anaconda-client -y
```

O `source` no início é obrigatório em shells não-interativos (scripts,
containers) onde o shell não foi inicializado com `conda init`.

### Token do Anaconda Cloud com permissão `write` em `ufcg-ibm`

Gere em [anaconda.org/settings/access](https://anaconda.org/settings/access).
Marque os escopos `conda` e `write`. Guarde o token — ele só é mostrado
uma vez. Você precisa de acesso de escrita à organização `ufcg-ibm`
para publicar nesse canal.

### URL pública para o binário

`conda build` baixa o binário durante o build, então ele precisa estar
acessível via HTTPS. O padrão neste projeto é um **asset de release no
GitHub** — ver Passo 1.

---

## Passo 1 — Hospedar o Binário num Release do GitHub

`conda build` busca o binário a partir de uma URL declarada na receita.
O caminho mais limpo e rastreável é anexar o binário como asset de um
release do GitHub do projeto.

1. Escolha uma tag que reflita o build, ex.: `v2.21.0-gpu`.
2. No repo do GitHub: **Releases → Draft a new release**.
3. Selecione/crie a tag, preencha título e descrição e **arraste o
   binário** para a área de assets.
4. **Publish release.**

A URL de download que o conda vai usar é:
```
https://github.com/<org>/<repo>/releases/download/<tag>/<nome-do-asset>
```

O nome do asset que você sobe vira parte dessa URL e precisa permanecer
**estável durante toda a vida da versão** — a receita vai fixar esse
nome.

---

## Passo 2 — Escrever o `meta.yaml`

O `meta.yaml` é a receita: ele diz pro `conda build` o que baixar, como
instalar e quais dependências de runtime declarar para o pacote funcionar
na máquina do usuário.

Crie um diretório de receita (qualquer nome; uma receita por diretório):

```bash
mkdir ~/minha_receita
cd ~/minha_receita
```

Escreva o `meta.yaml`. Esqueleto para uma **wheel Python**:

```yaml
package:
  name: <nome-do-pacote>        # nome publico do pacote (ex.: tensorflow-gpu)
  version: "<versao>"           # string no estilo PEP 440; entre aspas pro YAML tratar como string

source:
  url: https://github.com/<org>/<repo>/releases/download/<tag>/<asset>

build:
  number: 0                     # incrementa em rebuilds da mesma versao (ver "Publicando uma Atualizacao")
  script: pip install --no-deps $SRC_DIR/<asset>

requirements:
  host:                         # disponivel DURANTE o build (em ambiente isolado)
    - python =<X.Y>             # fixa o interpretador — wheels Python sao atreladas ao ABI
    - pip
  run:                          # instalado JUNTO com o pacote na maquina do usuario
    - python >=<X.Y>,<<X.Y+1>
    # ... dependencias de runtime, cada uma disponivel no conda-forge ou ufcg-ibm

about:
  home: https://github.com/<org>/<repo>
  license: <SPDX-id>            # ex.: Apache-2.0, MIT, BSD-3-Clause
  summary: "<descricao em uma linha>"
```

Observações:

- **`source.url`** é a URL do asset do GitHub do Passo 1. O `conda build`
  baixa o arquivo em `$SRC_DIR` antes de rodar `build.script`.
- **`build.script`** é shell rodando dentro do ambiente de build. Para
  wheels, `pip install --no-deps` basta — `--no-deps` porque as
  dependências do lado conda são declaradas em `requirements.run`, não
  pelas do pip.
- **`requirements.run`** é o que popula a árvore de dependências mostrada
  ao usuário durante o `conda install`. Todo nome aqui precisa resolver
  no conda-forge (ou no seu canal). Pinne com folga pra não travar o
  solver do usuário.

Para um **binário ou tarball puro** (sem Python), substitua o
`build.script`:
```yaml
build:
  number: 0
  script: |
    mkdir -p $PREFIX/bin
    install -m 755 $SRC_DIR/<binario> $PREFIX/bin/
```
`$PREFIX` é a raiz do ambiente conda — tudo que você instala aí dentro
vira parte do pacote.

---

## Passo 3 — Build

`conda build` cria um ambiente de build isolado, executa a receita e
empacota tudo que ficou em `$PREFIX` num arquivo `.conda`.

```bash
cd ~/minha_receita
conda build purge       # limpa diretorios temporarios de runs anteriores — default seguro
conda build .           # constroi a receita do diretorio atual
```

O que acontece:

1. Um ambiente novo é criado com os pacotes de `requirements.host`.
2. O `source.url` é baixado em `$SRC_DIR` (com cache pra próxima vez).
3. O `build.script` roda; o que for instalado em `$PREFIX` é capturado.
4. O resultado é comprimido e salvo em:
   ```
   ~/miniforge3/conda-bld/<subdir>/<nome>-<versao>-py<XY>_<number>.conda
   ```
   `<subdir>` é a arquitetura alvo (`linux-ppc64le`, `linux-64`, ...).

Localize o arquivo:
```bash
find ~/miniforge3/conda-bld -name '<nome>-<versao>-*.conda'
```

Perto do fim do build, o `conda build` roda uma série de checagens
estáticas em cada biblioteca compartilhada: linking, prefixes, símbolos
faltando. Mensagens *informacionais* (INFO/WARNING) são normais e não
falham o build; revise apenas os *errors*.

---

## Passo 4 — Testar Localmente Antes de Subir

Pegue dependências de runtime quebradas aqui, onde o único custo são
alguns segundos. A flag `--use-local` diz pro conda olhar no diretório
de build local **além** dos canais listados.

```bash
conda create -n _testepkg python=<X.Y> -y
conda activate _testepkg
conda install -c conda-forge --use-local <nome>=<versao> -y

# Smoke test pra pacote Python:
python -c "import <modulo>; print(<modulo>.__version__)"
```

Se o install resolver e o import funcionar, a receita está correta.

---

## Passo 5 — Upload para `ufcg-ibm`

`anaconda upload` envia o artefato `.conda` para o anaconda.org. A flag
`-u ufcg-ibm` publica no canal da organização em vez do seu namespace
pessoal de usuário.

```bash
PKG=$(find ~/miniforge3/conda-bld -name '<nome>-<versao>-*.conda' | head -1)
chmod 644 "$PKG"                                # o uploader exige world-readable
anaconda -t <TOKEN> upload -u ufcg-ibm "$PKG"
```

Em caso de sucesso, o CLI imprime a URL onde o pacote agora vive:
```
https://anaconda.org/ufcg-ibm/<nome>
```

---

## Passo 6 — Verificar a Partir do Canal Público

Esse é o mesmo caminho de instalação que os usuários finais vão usar.
Fazer em um env limpo confirma que nada foi puxado por acidente do seu
cache de build local.

```bash
conda create -n _testecanal python=<X.Y> -y
conda activate _testecanal
conda install -c ufcg-ibm -c conda-forge <nome>=<versao> -y

conda list | grep <nome>      # a terceira coluna deve mostrar: ufcg-ibm
```

Para um pacote Python, complete com o mesmo smoke test do Passo 4.

---

## Publicando uma Atualização

Para um novo build da **mesma `<versao>`** (ex.: rebuild com receita
corrigida ou asset upstream substituído):

1. Substitua o asset do release no GitHub, **mantendo o mesmo nome de
   arquivo**.
2. Incremente `build:number:` no `meta.yaml` (`0` → `1` → `2` ...). Isso
   é o que diz pro solver do conda que o novo artefato deve ser
   preferido em relação ao antigo.
3. Limpe o source cache pra forçar o conda a re-baixar o asset (o conda
   chaveia o cache pelo nome de arquivo, não pelo conteúdo remoto):
   ```bash
   conda build purge
   rm -f ~/miniforge3/conda-bld/src_cache/<asset>
   ```
4. Re-execute os Passos 3 → 5. O nome do arquivo resultante vai refletir
   o novo build number (`..._1.conda`), então velho e novo coexistem no
   canal e os solvers automaticamente pegam o mais alto.

Para uma nova **`<versao>`**: bump em `version:`, reset de
`build:number:` para `0`, suba o novo asset numa tag de release nova e
rode os Passos 3 → 5 sem alteração.

---

## Segurança

- Trate o token do Anaconda e qualquer GitHub PAT como **senhas**: nunca
  commite no git; rotacione se aparecerem em histórico de shell, num
  script ou em transcript de chat.
- Prefira GitHub PATs fine-grained com escopo limitado a um único repo.
- Tokens por ambiente (`~/.condarc` ou variável de ambiente) são mais
  limpos do que colar a cada `anaconda upload`.

---

## Referência Rápida

```bash
# Setup unico
source ~/miniforge3/etc/profile.d/conda.sh && conda activate base
conda install conda-build anaconda-client -y

# Build e upload
cd <diretorio-receita>
conda build purge && conda build .
PKG=$(find ~/miniforge3/conda-bld -name '<nome>-<versao>-*.conda' | head -1)
anaconda -t <TOKEN> upload -u ufcg-ibm "$PKG"

# Verificacao
conda create -n _t python=<X.Y> -y && conda activate _t
conda install -c ufcg-ibm -c conda-forge <nome>=<versao> -y
```
