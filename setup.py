from setuptools import setup, find_packages

setup(
    name="multimodal-hemoglobin-estimation",
    version="2.0.0",
    description="Multimodal Deep Learning for Non-Invasive Hemoglobin Estimation from Fingernail, Skin, and Conjunctival Images",
    author="Tanay Tonpe",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "streamlit>=1.35.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "Pillow>=10.0.0",
        "scikit-learn>=1.3.0",
        "joblib>=1.3.0",
        "matplotlib>=3.7.0",
        "openpyxl>=3.1.0",
        "kagglehub>=0.2.0",
    ],
)
