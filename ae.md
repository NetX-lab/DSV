# Artifact Evaluation

## 1. Requirements

### Hardware

- **Recommended:** NVIDIA H800 or H100 GPUs (at least two recommended for parallelism experiments)

### Software

- **Operating System:** Linux (Ubuntu 22.04 recommended)
- **CUDA:** Version 12.1
- **Deep Learning Frameworks:**
  - PyTorch 2.5.0
  - Triton 3.1.0

---

## 2. Installation

If you are using your own server, please follow these steps:

1. **Set up the Conda environment:**
   
   ```
   conda create -n asplos_ae python=3.10
   conda activate asplos_ae
   pip install -r requirements.txt
   ```

2. **Install the project:**
   
   ```
   cd DSV
   pip install -e .
   ```

3. **Compile the optimization solver:**
   
   ```
   cd optim
   make
   ```

---

## 3. Reproducing Main Results

The artifact contains two main components:

- **Released attention sparsity traces** (see the Observation section in the paper)
- **Main results reproduction:** Efficiency evaluation of our *video sparse attention* method and *sparsity-aware context parallelism*

---

### 3.1 Data Analysis

1. **Download the released dataset from Hugging Face:**
   
   ```
   cd sparsity_trace_analysis
   python download_script.py
   ```

2. **Run the analysis notebook:**  
   Open `analysis.ipynb` to visualize and explore the results. These scripts reproduce the main observations reported in the paper.  
   *Note: Due to storage constraints, only a sampled subset of full attention scores is provided.*

---

### 3.2 Efficiency Evaluation

> **Attention** is the main bottleneck for large video input diffusion models. While we cannot provide our in-house large-scale testbed (as full evaluations may take days and require terabytes of data), you can reproduce the key trends using the provided attention/parallelism modules and end-to-end evaluation at various sparsity levels.  
> All experiments are configured for 4 GPUs and tested on 4 H100 GPUs.

*Each experiment should complete within several minutes.*

#### 3.2.1 Video Sparse Attention

| Experiment                                                                                              | Script | Expected Output                                                                                     | Notes |
| ------------------------------------------------------------------------------------------------------- | ------ | --------------------------------------------------------------------------------------------------- | ----- |
| Low-rank overhead & sparse attention vs. full attention <br>(correctness check & forward/backward time) | —      | Correctness check passes (shown in standard output); a figure shows forward/backward time breakdown |       |
| Sparse attention speedup under different sparsity levels                                                | —      | Figure shows the speedup in forward and backward time for sparse attention                          |       |

#### 3.2.2 Parallelism

> *Note: Random indices are generated in these tests, which may affect computation/communication patterns and absolute timings. However, the overall trend is consistent.  
> Since experiments are mainly configured for 4 GPUs, the naive SCP's allgather communication inefficiency is less pronounced than with 8 GPUs.*

| Experiment                                                            | Script | Expected Output                                                                                         | Notes |
| --------------------------------------------------------------------- | ------ | ------------------------------------------------------------------------------------------------------- | ----- |
| Sparse KV gather correctness in sparsity-aware SCP                    | —      | Correctness check passes for sparse KV gather (forward & backward) shown in standard output             |       |
| Sparsity-aware HCP vs. naive HCP (computation imbalance optimization) | —      | Standard output and a figure show that naive HCP has computation stragglers and longer computation time |       |
| Sparsity-aware SCP vs. naive SCP (communication optimization)         | —      | Standard output and a figure show that naive SCP has longer communication time                          |       |
| End-to-end comparison                                                 | —      | Two figures for two cases, comparing sparsity-aware CP with naive HCP and naive SCP                     |       |

#### 3.3.3 End-to-end Throughput

To be updated.

---
