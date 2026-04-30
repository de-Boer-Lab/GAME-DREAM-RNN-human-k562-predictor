# GAME DREAM-RNN Human K562 Predictor

DREAM-RNN model [(Rafi et al. 2024, Nature Biotechnology)](https://www.nature.com/articles/s41587-024-02414-w) trained on MPRA data from [Agarwal et al. 2025](https://www.nature.com/articles/s41586-024-08430-9) for sequence-to-expression prediction (single task, K562 only).

**The Predictor:**

    - Returns `point` expression predictions only.
    - Always predicts in K562 regardless of the `cell_type` requested (Matcher is not used; `cell_type_actual` is always `"K562"` in responses).
    - Adds its own hardcoded 15bp upstream and 15bp downstream adapters internally -- these are part of the trained model's input and are not configurable.
    - Reduces incoming requests to the model's 200bp input window using a multi-branch preprocessing pipeline (see [Section 2.2](#22-preprocessing-logic)).

## Important Links

- To learn more about the GAME Framework ([Main GAME Repository](https://github.com/de-Boer-Lab/Genomic-API-for-Model-Evaluation), [preprint](https://www.biorxiv.org/content/10.1101/2025.07.04.663250v1.full))
- GAME Documentation: [ReadTheDocs](https://genomic-api-for-model-evaluation-documentation.readthedocs.io)
- Pre-built DREAM-RNN container image: [Zenodo](<<ADD NEW LINK HERE>>)
- To learn more about DREAM-RNN: [DREAM-RNN Human K562](https://github.com/de-Boer-Lab/random-promoter-dream-challenge-2022/tree/main/benchmarks/human)

---

## **Overview**

This document outlines the structure of the API codebase for DREAM-RNN and how it integrates with the containerized setup. The architecture is designed as a containerized microservice that communicates via HTTP REST endpoints.

---

## **1. DREAM-RNN API Structure**

The DREAM-RNN API is organized as follows. The Predictor uses a Flask application script and separates logic into distinct utility folders:

```bash
DREAM_RNN_GAME/
├── README.md
├── dream_rnn_environment.yml                # Conda environment file
└── src
    ├── README.md
    ├── dream_rnn_predictor.def              # Predictor container definition file
    ├── dream_rnn_script_and_utils/          # Model-specific logic
    │   ├── dreamRNN_predict.py              # Core model inference script
    │   ├── dream_rnn_preprocessing_utils.py # Preprocessing utilities
    │   ├── dream_rnn_k562_model_weight/     # Pre-trained model weights
    │   │   ├── model_best.pth
    │   │   └── ... (optimizer, scheduler, metrics)
    │   └── prixfixe/                        # Model framework scripts
    └── script_and_utils/                    # API and Server logic
        ├── config.py                        # Predictor name + paths + wire formats
        ├── dream_rnn_predictor_rest_api.py  # Flask-based REST API script
        ├── error_checking_functions.py      # Error classes and request validation
        ├── predictor_content_handler.py     # Request/response (de)serialization
        ├── predictor_help_message.json      # Help message file
        └── schema_validation.py             # Schema validation + preprocessing entry
```

---

## **2. Understanding the API**

### **Predictor API (Server)**

- **Purpose**: A Flask-based web server that listens for HTTP requests, validates inputs, runs the DREAM-RNN model, and returns structured JSON responses.
- **Core Script**: `src/script_and_utils/dream_rnn_predictor_rest_api.py`.
- **Supporting Scripts**:
  - `config.py` &mdash; predictor name, paths, supported wire formats
  - `schema_validation.py` &mdash; schema validation entry + preprocessing dispatch
  - `predictor_content_handler.py` &mdash; request decode / response encode
  - `error_checking_functions.py` &mdash; error classes (`BadRequestError`, `PredictionFailedError`, `ServerError`) and per-field validators
  - `dream_rnn_preprocessing_utils.py` &mdash; sequence preprocessing pipeline
  - `dreamRNN_predict.py` &mdash; model inference
  - `predictor_help_message.json` &mdash; help endpoint contents
- [PrixFixe](https://github.com/de-Boer-Lab/random-promoter-dream-challenge-2022/blob/main/prixfixe/readme.MD) Framework &mdash; the underlying neural network architecture (BHI first/core blocks + Autosome final block).

### **2.1 Key Features**
 
1. **Versioned Predictor Name**:
    - Inside the container, the predictor name is auto-versioned with the Apptainer build timestamp: `DREAM-RNN_Human_K562_YYYYMMDD-HHMMSS_TZ` (e.g. `DREAM-RNN_Human_K562_20260407-140628_PDT`).
    - Outside the container (dev mode), it falls back to `DREAM-RNN_Human_K562_dev`.
    - The build timestamp is read from `/.singularity.d/labels.json` at startup and embedded in every response (`predictor_name` field). This makes it possible to trace evaluation results back to the exact container build that produced them.
2. **Dynamic Path Handling**:
    - `config.py` uses `os.path.exists('/.singularity.d')` to determine whether the script is running inside the container.
    - Adjusts `DREAM_DIR` (model + preprocessing utilities) and `HELP_FILE` accordingly.
3. **HTTP Server Setup**:
    - **Flask Application**: Initializes a Flask web application that listens on a configurable `HOST` and `PORT` for incoming API requests.
    - **API Endpoints**: Exposes dedicated endpoints: `POST /predict` for inference, `GET /help` for metadata, and `GET /formats` for supported wire formats.
    - **Wire Format**: JSON only (`application/json` for both request and response).
4. **Request Validation**:
    - Validates mandatory top-level keys (`readout`, `prediction_tasks`, `sequences`) and per-task keys (`name`, `type`, `cell_type`, `species`).
    - Rejects unsupported request features at the API edge before model inference (see [Section 2.3](#23-supported-request-features)).
    - Returns standardized error responses (400, 422, 500) keyed by the violation category.
5. **Prediction Workflow**:
    - Decodes the request, validates schema, preprocesses sequences (see [Section 2.2](#22-preprocessing-logic)).
    - Calls `predict_dream_rnn(sequences, prediction_ranges, upstream_seq, downstream_seq, include_rev=True)`.
    - Assembles a structured JSON response with metadata (`type_actual`, `cell_type_actual`, etc.) and predictions per task.
    - Applies scaling: model output is natively log2; if `scale: linear` is requested, predictions are exponentiated (`2^x`).
6. **Help Endpoint**:
    - Returns the contents of `predictor_help_message.json` when `/help` is queried.

### **2.2 Preprocessing Logic**

For each sequence the predictor receives, the input is reduced to a 200bp probe before adapters are added:

```text
1. If constant flanks (upstream_seq / downstream_seq) provided:
   1.1. probe length <= target_length (200bp):
        Pad with biological flank context (NOT Ns)
          - Take tail of upstream_seq + probe + head of downstream_seq
          - If flanks too short to fill target_length, fall back to N-padding
        prediction_ranges IGNORED
   1.2. probe length > target_length:
        Center-crop probe to target_length
        prediction_ranges IGNORED

2. If no constant flanks:
   2.1. probe length <= target_length:
        Use the probe as-is, left-pad with Ns to target_length
        prediction_ranges IGNORED (probe is already the data we have)
   2.2. probe length > target_length AND prediction_ranges provided:
        Anchor target_length window MPRA_PROBE_TO_TSS_OFFSET (145bp) upstream
        of pr_start (TSS).
          start = max(0, pr_start - target_length - 145)
          end   = min(start + target_length, len(seq))
        Clamping start to 0 pulls a full target_length window of real sequence
        rather than synthesizing Ns to preserve a precise TSS offset-- DREAM-RNN
        has no explicit TSS-distance feature, so real sequence is strictly
        more useful than Ns at a specific offset.
        Left-pad with Ns only if the sequence itself is shorter than target_length.
   2.3. probe length > target_length AND no prediction_ranges:
        Center-crop to target_length

3. Add hardcoded DREAM-RNN adapters (15bp upstream + 15bp downstream)
4. One-hot encode the final sequence
```
 
**Key constants** (defined in `dream_rnn_preprocessing_utils.py` and `dreamRNN_predict.py`):
 
- `MPRA_PROBE_TO_TSS_OFFSET = 145` &mdash; assay-specific offset; in Agarwal-style lentiMPRA constructs, the regulatory probe ends exactly 145bp upstream of the EGFP TSS.
- `TARGET_LENGTH = 200` &mdash; model input length before adapters.
- `upstream_adapter_seq = "AGGACCGGATCAACT"` (15bp).
- `downstream_adapter_seq = "CATTGCGTGAACCGA"` (15bp).
### **2.3 Supported Request Features**
 
| Feature | Support | Notes |
|---|---|---|
| `readout = point` | ✅ | Only supported readout |
| `readout = track` / `interaction_matrix` | ❌ | Rejected with `bad_prediction_request` (400) |
| `type = expression` | ✅ | |
| `type = expression_*` (e.g. `expression_mRNA`, `expression_pol2`) | ✅ | All `expression_*` subtypes accepted |
| `type = binding_*`, `conformation_*`, `accessibility` | ❌ | Rejected with `bad_prediction_request` (400) |
| `species = homo_sapiens` | ✅ | |
| Other species | ❌ | Rejected with `prediction_request_failed` (422) |
| `scale = log` (default) | ✅ | Model output is natively log2 |
| `scale = linear` | ✅ | Output is exponentiated (`2^x`) |
| `cell_type` | Logged but ignored | Always returns K562 predictions |
| `upstream_seq` / `downstream_seq` | ✅ | Used as biological flank context for short probes |
| `prediction_ranges` | ✅ (conditional) | Used in Branch 2.2 of preprocessing only &mdash; ignored when flanks are provided or probes ≤ 200bp |

**Multi-task request behavior**: DREAM-RNN currently rejects an entire request if *any* task has an unsupported type, species, or scale. No model inference runs in this case &mdash; the request fails at validation before any compute is spent.

### **2.4 Error Handling**

| Status | Error key | Triggers |
|---|---|---|
| 400 | `bad_prediction_request` | Malformed JSON, missing mandatory keys, unsupported readout/type, invalid prediction_ranges format |
| 422 | `prediction_request_failed` | Sequence contains invalid characters, empty sequence, unsupported species/scale, prediction_range out of bounds |
| 500 | `server_error` | Model load failure, unexpected internal error |

---

---

## Interlude: Creating a Wrapper Function for API JSON Structure

Before diving into the configuration and running of the API, it is important to ensure that the output of DREAM-RNN (the raw predictions) is structured in compliance with the API JSON format.

The Predictor API relies on the model's output as the foundation for constructing its JSON response. The integration process becomes highly efficient as long as the model adheres to the `seq_id: expression_prediction` format, which DREAM-RNN does already.

---

### **Structure of the API JSON Format**

The API JSON format wraps predictions with metadata to describe tasks, cell types, and scaling. An example prediction JSON structure:

```json
{
    "predictor_name": "DREAM-RNN_Human_K562_20260407-140628_PDT",
    "prediction_tasks": [
        {
            "name": "task1",
            "type_requested": "expression",
            "type_actual": ["expression"],
            "cell_type_requested": "HEPG2",
            "cell_type_actual": "K562",
            "scale_prediction_requested": "log",
            "scale_prediction_actual": "log",
            "species_requested": "homo_sapiens",
            "species_actual": "homo_sapiens",
            "predictions": {
                "seq1": 0.3173099458217621,
                "seq2": 0.33908841013908386,
                "random_seq": 0.37649109959602356,
                "enhancer": 0.37649109959602356,
                "control": 0.37649109959602356
            }
        }
    ]
}
```

### Why Structure Predictions?

- **Standardization**: Ensures predictions are formatted with metadata and nested appropriately for the API.
- **Compatibility**: Bridges the gap between raw predictions and the API's required JSON format.
- **Dynamic Handling**: Allows the system to scale across diverse input sequences and multiple prediction tasks, enhancing flexibility.

The wrapper allows for predictions to be made by the model when the `predict_dream_rnn(sequences, include_rev=True)` is called by the Predictor API, returning predictions in a dictionary format, such that they can just be appended into the response as per the API JSON schema.

---
---

## **3. Configuring and Running the API** (Similar to [Basic Instructions for Test Evaluator and Predictor](https://github.com/de-Boer-Lab/Genomic-API-for-Model-Evaluation/tree/main/src/training_examples/Apptainer/Test_Evaluator_Predictor))

### 3.1 Configuring the Containers Using Definition Files

Containerizing the DREAM-RNN API involves creating **definition (.def) files** that specify the structure, dependencies, and environment of the container. This ensures a consistent and reproducible environment for running the Predictor and Evaluator APIs.

### **Purpose of [Definition Files](https://apptainer.org/docs/user/latest/definition_files.html)**

Definition files provide a declarative way to define:

1. **Base Image**: The starting point of the container (e.g. ubuntu or python:3.13-slim).
2. **File Structure**: Which files to copy into the container during build and which directories to bind/mount at runtime.
3. **Dependencies**: System-level and Python libraries required for the API to function.
4. **Execution Environment**: Configurations such as environment variables, Python environments, and permissions.
5. **Entry point**: Specifies the script to run when the container starts.

### Why Containers for APIs?

1. **Reproducibility**: Containers guarantee the same environment across different systems.
2. **Modularity**: Encapsulates all dependencies, scripts, and configurations in a single container.
3. **Portability**: Allows seamless deployment on various platforms, whether it is local or HPC clusters.

### **Why These Specific Configurations?**

#### **Predictor API Container**

1. **Base Image**:
    - `python:3.13-slim` provides a minimal Debian base. The container installs Miniconda and creates a `dream-rest` conda environment from `dream_rnn_environment.yml`, which is what the predictor actually runs against — the bootstrap Python is unused at runtime.
2. **File Inclusions**:
    - **Core API Scripts** in `/script_and_utils/`: `config.py`, `dream_rnn_predictor_rest_api.py`, `error_checking_functions.py`, `predictor_content_handler.py`, `schema_validation.py`, `predictor_help_message.json`.
    - **Model-Related Files** in `/dream_rnn_script_and_utils/`: `dreamRNN_predict.py`, `dream_rnn_preprocessing_utils.py`, the pre-trained weights (`model_best.pth`), and the [PrixFixe](https://github.com/de-Boer-Lab/random-promoter-dream-challenge-2022/blob/main/prixfixe/readme.MD) framework directory.
3. **Environment Variables**:
    - `PATH` and `LD_LIBRARY_PATH` ensure the `dream-rest` Conda environment is used during runtime.
    - `APPTAINER_NO_MOUNT="home,tmp,proc,sys,dev"` to provide a safe default that prevents common host directories from being automatically mounted.
4. **Runtime Needs**:
    - No mounting of input data is required — the Predictor is self-contained (model weights and code are baked into the container). The only inputs come over HTTP from the Evaluator.

---

### **3.2. Best Practices for Container Isolation**
 
For maximum security and reproducibility, run the container with the `--containall` flag.
 
This flag creates a strictly isolated environment, preventing the container from accessing host files (like `/home`, `/tmp`) or external environment variables. This practice is critical for reproducible runs as it ensures the container execution is not accidentally influenced by the host system.
 
**Layered Isolation Approach**:
 
- **Definition file (`.def`)**: The container is built with `APPTAINER_NO_MOUNT` configurations to provide a safe default. This setting prevents common host directories from being automatically mounted, making the container inherently more isolated by design.
- **Runtime (`--containall`)**: While defaults are useful, the `--containall` flag provides complete, enforced isolation at runtime. It blocks all unexpected host directories, scrubs environment variables, and creates separate IPC/PID namespaces.
**Note:** The DREAM-RNN Predictor is fully self-contained and does not require `-B` bind mounts — all model weights and scripts live inside the container.

---

### **3.3. Running the Predictor API**

1. **Start the Predictor API container**
    The Predictor API must be running first since it listens for incoming connections from the Evaluator. Use the following command:
    ```bash
    apptainer run --nv --containall dream_rnn_predictor.sif HOST PORT
    ```
 
    - `--nv`: Enables NVIDIA GPU support (recommended for efficient DREAM-RNN inference; falls back to CPU if no GPU is available).
    - Replace `HOST` with the server's IP, which can be found using `hostname` followed by a `-i` or `-I` flag, and `PORT` with the desired [port number](https://www.geeksforgeeks.org/50-common-ports-you-should-know/).
    - The Predictor container will bind to the specified host and port and expose the `/predict`, `/help`, and `/formats` endpoints.
    - Ensure that the port (e.g., `5000`) is open and not blocked by any firewall or network policies. Ports above 1024 are usually free to use on most computers/servers.
2. **Validate the server**
    - Ensure it listens on the specified host and port.
    - On startup you should see something like:

    ```bash
    DREAM-RNN_Human_K562_20260407-140628_PDT Predictor is running on http://172.16.47.244:5000
    * Running on http://172.16.47.244:5000/ (Press CTRL+C to quit)
    ```

    The first line confirms both the versioned predictor name (which will appear in every response) and the listening address.

---

## **4. Example Input and Output JSON Files**

### **4.1. Input JSON**

- Located in the `evaluator_data/` directory.
- Example structure:

    ```json
    {
    "readout": "point",
    "prediction_tasks": [
        {
        "name": "gosai_synthetic_sequences",
        "type": "expression",
        "cell_type": "K562",
        "scale": "linear",
        "species": "homo_sapiens"
        }
    ],
    "sequences":{
        "7:70038969:G:T:A:wC": "CCTGGTCTTTCTTGCTAAATAAACATATCGTGCATCATCCAGATCTTGCTGAAATTTGGGGGATATGCATTGAAGCAGCCCCTGTTTCTCCATGAAGGTTTATGTCTGTGAGCCTGGCTGTGCAGTTGGGAGGCCTGGGGGAGAGGTCATGCTTCTACCATGGCGTTTTCCATTTTCCTTAAAATGTGCCTCAGCAACAG",
        "1:192696196:C:T:A:wC": "CATAAAGATGAGGCTTGGCAAAGAACATCTCTCGGTGCCTCCCATTTCATTGTCCCTAAAGTAGAAGCTGAGTGTCATCATTTGTTAAAATTGGGGAAGTCTCCGAGGTGTGGGTTCATCAGAACAATAGCCACTGTTGCCTGTGGTCACAGTCACTGAAGCTGGGGTCCTGGTCACTACTCCAACAGCTGGGAGGCAGC",
        "1:211209457:C:T:A:wC": "CATAAAGCCAATCACTGAGATGACAAGTACTGCCAGGAAAGAAGGCTTTAATCGGGTATTGCAGCTGAAGAGATAGGAGAGCAGTCTCAAATCCATCTCTCTGACCAACTAAAATTGGGGGTTTATGTAGTGGGGAAGGAATGTAGCTACATGTGGGTAAACAGGAATTAGGGAGGGGTAGGGAAGAAGAGTTGGCCATC",
        "15:89574440:GT:G:A:wC": "CATAAAGGCAGTGTAGACCCAAACAGTGAGCAGTAGCAAGATTTATTACAAAGAGCGAAAGAAGAACGAAACCACATCGCAAAACGGAACTCCAGCCGGTTGCCACTACTGCCTCGGGCAGCCTGCTTTTATTCTCTTATCTGGCCCCACCCACATCCTGCTGATTGGTCCATTTTACAGAGAGTGGATTGGTCCATTT",
        "15:89574440:GT:G:R:wC": "CATAAAGGCAGTGTAGACCCAAACAGTGAGCAGTAGCAAGATTTATTACAAAGAGCGAAAGAAGAACGAAACCACATCGCAAAACGGAACTCCAGCCGGTTTGCCACTACTGCCTCGGGCAGCCTGCTTTTATTCTCTTATCTGGCCCCACCCACATCCTGCTGATTGGTCCATTTTACAGAGAGTGGATTGGTCCATTT"
        }
    }
    
    ```

### **4.2. Output JSON**

- Saved in the `predictions/` directory.
- Example structure:

    ```json
    {
    "predictor_name": "DREAM-RNN_Human_K562_20260407-140628_PDT",
    "prediction_tasks": [
        {
            "name": "gosai_synthetic_sequences",
            "type_requested": "expression",
            "type_actual": ["expression"],
            "cell_type_requested": "K562",
            "cell_type_actual": "K562",
            "scale_prediction_requested": "linear",
            "scale_prediction_actual": "linear",
            "species_requested": "homo_sapiens",
            "species_actual": "homo_sapiens",
            "predictions": {
                "7:70038969:G:T:A:wC": -0.4900762140750885,
                "1:192696196:C:T:A:wC": -0.42054876685142517,
                "1:211209457:C:T:A:wC": -0.251442551612854,
                "15:89574440:GT:G:A:wC": 1.1541708707809448,
                "15:89574440:GT:G:R:wC": 1.1637296676635742
                }
            }
        ]
    }
    ```

The `predictor_name` in the response matches the container build timestamp, allowing evaluation results to be traced back to a specific build.
