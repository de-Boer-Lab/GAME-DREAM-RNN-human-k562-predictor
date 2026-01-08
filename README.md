# GAME-DREAM-RNN-human-k562-predictor

DREAM-RNN model [(Rafi et al. 2024, Nature Biotechnology)](https://www.nature.com/articles/s41587-024-02414-w) trained on MPRA data from Agarwal et. al 2023 and can be used for any sequence-to-expression predictions (single task).

The DREAM-RNN Predictor can only return point expression predictions for K562 (Matcher is never used, and it will return K562 predictions regardless of the cell type requested). Because it has hardcoded adapters (that were present in the model’s training data), it will ignore any adapter sequences that are sent. Sequences shorter than its 200bp input are centered and padded equally with Ns on either side, while sequence longer than 200bp are cropped to the target input length. Prediction ranges sent from the Evaluator are used to crop the input sequence to the desired start and stop indices.

## Important Links

- [Main GAME Repository](https://github.com/de-Boer-Lab/Genomic-API-for-Model-Evaluation)
- GAME Documentation: <LINK HERE!!!>
- [DREAM-RNN K562](https://github.com/de-Boer-Lab/random-promoter-dream-challenge-2022/tree/main/benchmarks/human)

---

---

## **Overview**

This document outlines the structure of the API codebase for DREAM-RNN and how it integrates with the containerized setup. The architecture is designed as a containerized microservice that communicates via HTTP REST endpoints.

---

---

## **1. DREAM-RNN API Structure**

The DREAM-RNN-API is organized as follows. The Predictor now utilizes a Flask application script and separates logic into distinct utility folders.

```bash
DREAM_RNN_GAME/
├── README.md
├── dream_rnn_environment.yml          # Conda environment file
└── src
    ├── README.md
    ├── predictor.def                  # Predictor container definition file
    ├── dream_rnn_script_and_utils/    # Model-specific logic
    │   ├── dreamRNN_predict.py             # Core model inference script
    │   ├── dream_rnn_preprocessing_utils.py # Preprocessing utilities
    │   ├── dream_rnn_k562_model_weight/    # Pre-trained model weights
    │   │   ├── model_best.pth
    │   │   └── ... (optimizer, scheduler, metrics)
    │   └── prixfixe/                       # Model framework scripts
    └── script_and_utils/                   # API and Server logic
        ├── dream_rnn_predictor_rest_api.py # Flask-based REST API script
        ├── error_checking_functions.py     # Error handling
        ├── predictor_content_handler.py    # Request processing logic
        ├── predictor_help_message.json     # Help message file
        └── schema_validation.py            # Input JSON validation
```

---

---

## **2. Understanding the API**

### **Predictor API (Server)**

- **Purpose**: A Flask-based web server that listens for HTTP requests, validates inputs, runs the DREAM-RNN model, and returns structured JSON responses.
- **Core Script**: `src/script_and_utils/dream_rnn_predictor_rest_api.py`.
- **Supporting Scripts, Error Handling, Help Files**:
  - `schema_validation.py`
  - `predictor_content_handler.py`
  - `dream_rnn_preprocessing_utils.py`
  - `dreamRNN_predict.py`
  - `error_checking_functions.py`
  - `predictor_help_message.json`.
- [PrixFixe](https://github.com/de-Boer-Lab/random-promoter-dream-challenge-2022/blob/main/prixfixe/readme.MD) Framework.
- **Key Features**:
    1. **Dynamic Path Handling**:
        - Uses `os.path.exists` to determine if the script is running inside the container.
        - Adjusts the path to the DREAM-RNN model (`DREAM_DIR`) and helper files (`HELP_FILE`) accordingly.
    2. **HTTP Server Setup**:
        - **Flask Application**: Initializes a Flask web application that listens on a configurable `HOST` and `PORT` for incoming API requests.
        - **API Endpoints**: Exposes dedicated endpoints for interaction: `POST /predict` for main inference, `GET /help` for metadata, and `GET /formats` for supported data types.
        - **Standard Protocol**: Utilizes standard HTTP/1.1 for reliable data transmission, ensuring compatibility with standard HTTP clients.
    3. **Request Validation**:
        - Validates mandatory keys (`request`, `prediction_tasks`, etc.) and values in the incoming JSON request using error-checking functions.
        - Returns error messages if any validation step fails.
    4. **Prediction Workflow**:
        - Extracts and validates sequences and prediction tasks.
        - Prepares and makes a function call to the DREAM-RNN model using `predict_dream_rnn`.
        - Generates a structured JSON response with:
            - Metadata (`type_actual`, `cell_type_actual`, etc.)
            - Prediction results
    5. **Dynamic Help Handling**:
        - Returns a help message when the `/help` endpoint is accessed, loading information from the `HELP FILE`.
    6. **Data Transfer**:
        - Sends error messages, help content, or predictions back to the Evaluator API as standard JSON HTTP responses.

---

---

## Interlude: Creating a Wrapper Function for API JSON Structure

Before diving into the configuration and running of the API, it is important to ensure that the output of DREAM-RNN (the raw predictions) are structured in compliance with the API's JSON format.

The Predictor API relies on the model's output as the foundation for constructing its JSON response. The integration process becomes highly efficient as long as the model adheres to the `seq_id: expression_prediction` format, which DREAM-RNN does already.

---

---

### **Structure of the API JSON Format**

The API JSON format wraps predictions with metadata to describe tasks, cell types, and scaling. An example prediction JSON structure:

```json
{
    "predictor_name": "DREAM-RNN_Human_K562",
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

1. **Base Image**: The starting point of the container (e.g. ubuntu or python:3.9-slim).
2. **File Structure**: Which files to copy into the container during build and which directories to bind/mount at runtime.
3. **Dependencies**: System-level and Python libraries required for the API to function.
4. **Execution Environment**: Configurations such as environment variables, Python environments, and permissions.
5. **Entry point**: Specifies the script to run when the container starts.

### Why Containers for APIs?

1. **Reproducibility**: Containers guarantee the same environment across different systems.
2. **Modularity**: Encapsulates all dependencies, scripts, and configurations in a single container.
3. **Portability**: Allows seamless deployment on various platforms, whether it is local or HPC clusters.

### **Why These Specific Configurations?**

### **Predictor API Container**

1. **Base Image**:
    - `python:3.13-slim` is chosen for its lightweight nature and Python-specific optimizations.
    - This base image minimizes overhead while supporting Python dependencies.
2. **File Inclusions**:
    - **Core Script**: `dream_rnn_predictor_rest_api.py`, responsible for handling HTTP requests
    - **Model-Related Files**:
        - `dreamRNN_predict.py`: Handles model loading and predictions.
        - Pre-trained model weights (`model_best.pth`).
    - **Supporting Scripts, Error Handling, Help Files**:
        - `dream_rnn_preprocessing_utils.py` for general sequence preprocessing functions
        - `error_checking_functions.py` for error handling
        - `predictor_content_handler.py` for serialization support
        - `schema_validation.py` for validating incoming JSON request payloads against the API schema
        - `predictor_help_message.json` to provide metadata about the API
    - [PrixFixe](https://github.com/de-Boer-Lab/random-promoter-dream-challenge-2022/blob/main/prixfixe/readme.MD) **Framework**: Essential for the DREAM-RNN prediction pipeline.
3. **Environment Variables**:
    - `PATH` and `LD_LIBRARY_PATH` ensure the `dream` Conda environment is used during runtime, isolating dependencies required for prediction tasks.
4. **Runtime Needs**:
    - No mounting is required as the Predictor container is pre-packaged with all scripts, dependencies, and data files.
    - This design simplifies deployment, making the container self-contained and portable.

---

### **3.2. Best Practices for Container Isolation**

For maximum security and reproducibility, it is highly recommended to run both containers with the `--containall` flag.

This flag creates a strictly isolated environment, preventing the container from accessing host files (like `/home`, `/tmp`) or external environment variables. This practice is critical for reproducible run as it ensures the container execution is not accidentally influenced by the host system.

**Layered Isolation Approach**
    - Definition file (`.def`): The containers are built with `APPTAINER_NO_MOUNT` configurations to provide a safe default. This setting is intended to prevent common host directories from being automatically mounted, making the container inherently more isolated by design.
    - Runtime (`--containall`): While defaults are useful, the `--containall` flag provides complete, enforced isolation at runtime. It blocks all unexpected host directories, scrubs environment variables, and creates separate IPC/PID namespaces. This is the best practice to guarantee a run is entirely isolated.

**Note:** Because `--containall` blocks access to host files by default, the `-B` (bind) flag must be used to explicitly allow the container to read inputs and write outputs.

```bash
# -B is required to give access to specific folders when using --containall
apptainer run --containall -B /local/data:/data my_container.sif ...
```

---

### **3.3. Running the Predictor API**

1. **Start the Predictor API container**

    The Predictor API must be running first since it listens for incoming connections from the Evaluator. Use the following command to start the Predictor:

    ```bash
    apptainer run --nv --containall dream_rnn_predictor.sif HOST PORT
    ```

    - `-nv`: Enables NVIDIA GPU support (required for efficient DREAM-RNN inference).
    - Replace `HOST` with the server's IP, which can be found using `hostname -I`, and `PORT` with the desired [port number](https://www.geeksforgeeks.org/50-common-ports-you-should-know/).
    - The Predictor container will bind to the specified host and port and expose endpoints.
    - Ensure that the port (e.g., `5000`) is open and not blocked by any firewall or network policies. Ports above 1024 are usually free to use on most computers/servers.  

2. **Validate the server**
    - Ensure it listens on the specified host and port.
    - Example:

        ```bash
        * Running on http://172.16.47.244:5000/ (Press CTRL+C to quit)
        ```

---

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
    "predictor_name": "DREAM-RNN_Human_K562",
    "prediction_tasks": [
        {
            "name": "gosai_synthetic_sequences",
            "type_requested": "expression",
            "type_actual": ["expression"],
            "cell_type_requested": "K562",
            "cell_type_actual": "K562",
            "scale_prediction_requested": "linear",
            "scale_prediction_actual": "log",
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

---

---
