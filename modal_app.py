"""Modal deployment for ManGER - Manga Translation App."""
import modal

app = modal.App("manger")

# Define the container image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "libgl1",
        "libglib2.0-0", 
        "fonts-dejavu-core",
        "fonts-liberation",
        "fontconfig",
    )
    .pip_install(
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "transformers>=4.35.0",
        "timm>=0.9.0",
        "einops>=0.7.0",
        "pillow>=10.0.0",
        "opencv-python>=4.8.0",
        "numpy>=1.24.0",
        "streamlit>=1.28.0",
        "loguru>=0.7.0",
        "openai>=1.0.0",
        "httpx>=0.25.0",
        "pymupdf>=1.23.0",
        "beautifulsoup4>=4.12.0",
        "requests>=2.31.0",
        "pydantic>=2.0.0",
        "pydantic-settings>=2.0.0",
        "shapely>=2.0.0",
        "matplotlib>=3.7.0",
        "playwright>=1.40.0",
    )
    .run_commands("playwright install chromium && playwright install-deps chromium")
    .add_local_dir("src", "/app/src", copy=True)
    .add_local_dir("pages", "/app/pages", copy=True)
    .add_local_dir(".streamlit", "/app/.streamlit", copy=True)
    .add_local_file("app.py", "/app/app.py", copy=True)
    .add_local_file("pyproject.toml", "/app/pyproject.toml", copy=True)
    .add_local_file("README.md", "/app/README.md", copy=True)
    .run_commands("cd /app && pip install -e .")
)


@app.function(
    gpu="T4",  # Options: "T4", "A10G", "A100"
    image=image,
    timeout=3600,  # 1 hour max
    secrets=[modal.Secret.from_name("openai-secret")],  # Add your API key as a Modal secret
    max_containers=1,
)
def run_streamlit():
    import subprocess
    import os
    import modal
    
    os.chdir("/app")
    
    # Start Streamlit
    proc = subprocess.Popen([
        "streamlit", "run", "app.py",
        "--server.port=8501",
        "--server.address=0.0.0.0",
        "--server.enableCORS=false",
        "--server.enableXsrfProtection=false",
        "--server.maxUploadSize=500",
        "--server.enableWebsocketCompression=false",
        "--browser.gatherUsageStats=false",
        "--server.fileWatcherType=none",
    ])
    
    # Use modal.forward to expose port directly (bypasses proxy issues)
    with modal.forward(8501) as tunnel:
        print(f"Streamlit running at: {tunnel.url}")
        proc.wait()


# For local testing
@app.local_entrypoint()
def main():
    print("Deploying ManGER to Modal...")
    print("URL will be shown after deployment")
