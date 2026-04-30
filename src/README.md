# Configuring Definition File and Running Predictor Container for DREAM-RNN

## Overview

This container for the Predictor includes:

- Flask-based REST API (`dream_rnn_predictor_rest_api.py`) for sequence processing and error handling.
- Integrated DREAM-RNN model with its dependencies and `dream-rest` conda environment created using `dream_rnn_environment.yml`.
- Pre-trained model weights (`model_best.pth`) for predictions.
- Support scripts: `config.py`, `schema_validation.py`, `dream_rnn_preprocessing_utils.py`, `error_checking_functions.py`, `predictor_content_handler.py`, and `predictor_help_message.json`.

## Usage

We encourage using pre-built containers for this model that are hosted on Zenodo: <NEW_LINK_HERE>.

If you want to build your own Predictor, refer to the documentation on [Building your own GAME Modules](https://genomic-api-for-model-evaluation-documentation.readthedocs.io/en/latest/Building_Modules/).

### Run the container

```bash
apptainer run --nv --containall dream_rnn_predictor.sif HOST PORT
```

### Details

- The container launches a Flask-based server to receive data via HTTP POST requests.
- Replace `HOST` and `PORT` with the server and port configuration for the evaluator.
- The server prints its versioned predictor name on startup (e.g. `DREAM-RNN_Human_K562_20260407-140628_PDT`), which will appear in every response's `predictor_name` field for traceability.

### API Endpoints

- `GET  /formats` — Returns supported request and response formats
- `GET  /help`    — Returns predictor metadata and usage information
- `POST /predict` — Main endpoint for submitting prediction requests

### Purpose

- Facilitates genomic model evaluation and prediction using the DREAM-RNN framework.
- It is designed to seamlessly integrate with other tools via API endpoints.

### Example Command

```bash
apptainer run --nv --containall dream_rnn_predictor.sif 172.16.47.244 5000
```

### Arguments

1. `HOST`: IP address or hostname of the Predictor server.
2. `PORT`: Port number the Predictor is listening on.

## Additional Notes about the `%environment` Block in the Definition File

```bash
%environment
    # Prevent automatic binding of host directories
    export APPTAINER_NO_MOUNT="home,tmp,proc,sys,dev"
    export LC_ALL=C
    export PATH="/opt/conda/envs/dream-rest/bin:$PATH"
    export LD_LIBRARY_PATH="/opt/conda/envs/dream-rest/lib:$LD_LIBRARY_PATH"
```

- `export APPTAINER_NO_MOUNT="home,tmp,proc,sys,dev"`:
*Why it is required:* By default, Apptainer automatically mounts host directories (like `/home`, `/tmp`, `/proc`, `/sys`, and `/dev`) into the container. This can inadvertently expose host data or cause conflicts. Setting this variable disables those automatic mounts so that only explicitly bound directories (using the `-B` flag) will be available inside the container.

*However:* The `--containall` flag, used at runtime, provides the second and most complete layer of isolation by blocking all unexpected host directories, variables, and settings.

- `export LC_ALL=C`:
This sets the container to use the default "C" (POSIX) locale for consistent sorting, formatting, and error messages, regardless of the host's locale settings.

- `export PATH="/opt/conda/envs/dream-rest/bin:$PATH"`:
This modifies the `PATH` so that executables in the `dream-rest` Conda environment are prioritized. This is important for using the Python interpreter and other tools installed in that environment over any system defaults.

- `export LD_LIBRARY_PATH="/opt/conda/envs/dream-rest/lib:$LD_LIBRARY_PATH"`:
This ensures that the dynamic linker finds the libraries from the Conda environment first, which is crucial for using the correct versions of shared libraries.

### Additional Links for Reference

- [Apptainer Documentation](https://apptainer.org/docs/user/latest/)
- [HEP Software Foundation — Introduction to Apptainer/Singularity](https://hsf-training.github.io/hsf-training-singularity-webpage/)
