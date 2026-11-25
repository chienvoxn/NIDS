# Linux Deployment Guide

## Environment

The experimental system was tested on Ubuntu under WSL2
with Python 3.11 and CPU inference.

See the recorded Linux requirements freeze for exact versions.

## Activate existing environment

    source "$HOME/.venvs/nids/bin/activate"
    export PYTHONPATH="$PWD/src"
    export OMP_NUM_THREADS=2
    export MKL_NUM_THREADS=2

Run commands from the NIDS repository root.

## Model artifacts

CIC primary model:
outputs/final/cic_final_runs/seed_42/cnn_mhsa.pt

CTU primary model:
outputs/final/ctu_final_runs/seed_44/cnn_mhsa.pt

Use matching preprocessing and metrics artifacts.

Gaussian artifacts:
outputs/final/cic_final_runs/gaussian/
outputs/final/ctu_final_runs/gaussian/

## Check environment

    python -m pip check
    git lfs fsck

## Linux network interfaces

    ip link show
    sudo ip netns list

The recorded experiment uses the isolated nids-lab namespace
and the nids-veth0 interface.

The lab must already exist before running this example.

## Realtime experimental monitoring

    sudo env \
        PYTHONPATH="$PWD/src" \
        OMP_NUM_THREADS=2 \
        MKL_NUM_THREADS=2 \
        "$VIRTUAL_ENV/bin/python" \
        scripts/run_live_fusion.py \
        --iface nids-veth0 \
        --seconds 30 \
        --max-packets 10000 \
        --ack-unverified

Capture only authorized network traffic.

## Three-scenario replay

    python scripts/demo_attack_alerts.py \
        --scenario all \
        --limit 10 \
        --delay 0.1

The replay displays saved predictions from held-out data.
It does not generate network attacks.

## Experimental metrics

    python scripts/finalize_evaluation.py

## Botnet topology examples

    python scripts/evaluate_botnet_topology.py

## Regression tests

    python -m pytest -q tests test_cic_flow_extractor.py -rs

## Dashboard

After installing the dashboard dependencies:

    python -m streamlit run dashboard/final_app.py

Dashboard acceptance is a separate validation task.

## Safety and limitations

Realtime predictions are experimental and unverified.
No automatic blocking is performed.
See PROJECT_COMPLETION.md for research limitations.
