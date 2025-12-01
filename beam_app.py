"""Beam Cloud deployment for ManGER - Manga Translation App.

Features:
- Auto-standby after 10 min idle (no cost!)
- Permanent URL
- GPU support

Usage:
    pip install beam-client
    beam deploy beam_app.py:pod
"""
from beam import Image, Pod

# Use our existing Docker Hub image
image = Image(base_image="hugobart/manger:gpu")

# Define the Pod
pod = Pod(
    name="manger",
    image=image,
    gpu="A10G",  # Options: "T4", "A10G", "L4", "A100-40", "A100-80"
    cpu=2,
    memory="8Gi",
    ports=[8501],
    # Auto-standby: stops after 10 min without connections (default)
    # Set to -1 to keep alive indefinitely
    keep_warm_seconds=300,  # 5 minutes
    env={
        # Add your OpenAI key here or pass via CLI
        # "MANGER_TRANSLATE_OPENAI_API_KEY": "sk-...",
    },
    entrypoint=[
        "streamlit", "run", "/app/app.py",
        "--server.port=8501",
        "--server.address=0.0.0.0",
        "--server.enableCORS=false",
        "--server.enableXsrfProtection=false",
        "--server.maxUploadSize=500",
    ],
)
