# Artifact Evaluation

## 1. Requirements

### Hardware

- **Recommended:** 4× NVIDIA H800 or H100 80GB GPUs

### Software

- **Operating System:** Linux (Ubuntu 22.04 recommended)
- **CUDA:** 12.1
- **Deep Learning Frameworks:**
  - PyTorch 2.5.1
  - Triton 3.1.0

---

## 2. Installation

If using your own server, please follow these steps:

1. **Set up the Conda environment:**
   
   ```bash
   wget https://repo.anaconda.com/archive/Anaconda3-2025.06-0-Linux-x86_64.sh
   bash Anaconda3-2025.06-0-Linux-x86_64.sh # if anaconda not installed
   
   conda create -n asplos_ae python=3.11
   conda activate asplos_ae
   
   bash install_torch.sh # torch flashattn package
   pip install -r requirements.txt # other packages
   ```

2. **Install the project:**
   
   ```bash
   cd DSV
   pip install -e .
   ```

3. **Compile the optimization solver:**
   
   ```bash
   cd DSV/optim_solve
   make install
   ```

4. **Install Apex**
   
   ```bash
   git clone https://github.com/NVIDIA/apex
   # make sure cuda nvcc version match pytorch's 12.1
   cd apex
   # if pip >= 23.1 (ref: https://pip.pypa.io/en/stable/news/#v23-1) which supports multiple `--config-settings` with the same key... 
   pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
   # otherwise
   pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --global-option="--cpp_ext" --global-option="--cuda_ext" ./
   ```

---

## 3. Reproducing Main Results

This artifact includes:

- **Attention sparsity traces** (see *Observation* section in the paper)
- **Main results reproduction:** Efficiency evaluation of our *video sparse attention* and *sparsity-aware context parallelism*

---

### 3.1 Data Analysis

1. **Download the released dataset from Hugging Face (if needed):**
   
   ```
   cd sparsity_trace_analysis
   python download_script.py
   ```

2. **Run the analysis notebook:**  
   Open `analysis.ipynb` to visualize and explore results. These scripts reproduce the main observations reported in the paper. It mainly reproduces the results in `Figure 3,5,6,7`. 
   
   *Note: Only a sampled subset of full attention scores is provided due to storage constraints.*

---

### 3.2 Efficiency Evaluation

> **Attention** is the main bottleneck for large video input diffusion models.  
> While we cannot provide our in-house large-scale testbed (full evaluation may require days and terabytes of data), you can reproduce key trends using the provided modules and end-to-end evaluation at various sparsity levels.  
> All following experiments are configured for 4 GPUs (tested on 4 H100 GPUs).

*Each experiment should complete no more than 10 minutes. Scripts are under the `scripts` folder.*

#### 3.2.1 Video Sparse Attention

|      | Experiment                                                               | Command       | Expected Output                                                          | Notes                                                                                                                                             |
| ---- | ------------------------------------------------------------------------ | ------------- | ------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| exp1 | Low-rank overhead & sparse vs. full attention <br>(correctness & timing) | `bash run.sh` | Correctness check in stdout; figure with forward/backward time breakdown | Example logs and figures are available in the exp directory.<br/>Reference: Figure 19 in the paper for comparison (experimental settings differ). |
| exp2 | Sparse attention speedup across sparsity levels                          | `bash run.sh` | Figure showing speedup (forward/backward) for sparse attention           | Example logs and figures are available in the exp directory.<br/>Reference: Figure 18 in the paper for comparison (experimental settings differ). |

#### 3.2.2 Parallelism

> *Note: Random indices are generated in these tests, affecting computation/communication patterns and absolute timings, but trends are consistent.  
> **With 4 GPUs, naive SCP’s communication inefficiency is less pronounced than with 8 GPUs.**

|      | Experiment                                                            | Command       | Expected Output                                                                     | Notes                                                                                                                                             |
| ---- | --------------------------------------------------------------------- | ------------- | ----------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| exp3 | Sparse KV gather correctness in sparsity-aware SCP                    | `bash run.sh` | Correctness check passes (stdout)                                                   | Example logs are available in the exp directory.<br/>Stdout should show validation pass.                                                          |
| exp4 | Sparsity-aware HCP vs. naive HCP (computation imbalance optimization) | `bash run.sh` | Stdout and figure: naive HCP has computation stragglers and longer computation time | Example figure is available in the exp directory.<br/>Compare the computation time.                                                               |
| exp5 | Sparsity-aware SCP vs. naive SCP (communication optimization)         | `bash run.sh` | Stdout and figure: naive SCP has longer communication time                          | Example  figure is available in the exp directory.<br/>Compare the communication time.                                                            |
| exp6 | End-to-end time (comm+comp) comparison                                | `bash run.sh` | Two figures for two cases: comparing sparsity-aware CP with naive HCP and naive SCP | Example logs and figures are available in the exp directory.<br/>Reference: Figure 20 in the paper for comparison (experimental settings differ). |

#### 3.2.3 Model End-to-End Throughput

|      | Experiment                             | Command       | Expected Output                                                                         | Notes                                                                                                                                                       |
| ---- | -------------------------------------- | ------------- | --------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| exp7 | End-to-end model throughput comparison | `bash run.sh` | JSON files recording step times for each method; throughput comparison figure generated | Example JSON files and figures are available in the exp directory.<br/>Reference: Figure 16 in the paper for comparison (experimental settings may differ). |
