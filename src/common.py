import modal
app=modal.App("plantid")
vol=modal.Volume.from_name("plantid-vol",create_if_missing=True)
R="mikehemberger/plantnet300K"
cpu_img=modal.Image.debian_slim().pip_install("pyarrow","pandas","huggingface_hub","numpy","scikit-learn").add_local_python_source("common","probe")
gpu_img=modal.Image.debian_slim(python_version="3.11").pip_install("torch==2.5.1","torchvision==0.20.1","open_clip_torch==2.29.0","timm>=1.0.20","pyarrow","pandas","huggingface_hub","numpy","scikit-learn","pillow").add_local_python_source("common","probe")
