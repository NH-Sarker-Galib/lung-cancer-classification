\# Lung Cancer Classification from CT Scan Images



Deep learning-based lung cancer classification from CT scan images using EfficientNet-B3 and Swin Transformer, with GAN-based data augmentation and Grad-CAM++ explainability.



\## Overview



This project develops a deep learning pipeline for lung cancer classification using CT scan images.



The system classifies CT images into two categories:



\- Cancer

\- No Cancer



The project investigates the use of deep learning models, GAN-based synthetic data augmentation, and explainable AI techniques for lung cancer classification.



\## Research Objective



The main objective is to investigate whether quality-controlled synthetic CT image augmentation can improve lung cancer classification performance, particularly under limited-data conditions.



The study focuses on:



\- Classification accuracy

\- Sensitivity / Recall

\- F1-score

\- ROC-AUC

\- False-negative reduction

\- Comparison between CNN and Transformer architectures

\- Explainability of model predictions



\## Methodology



The overall pipeline consists of:



```text

CT Dataset

&#x20;   ↓

Data Inspection and Cleaning

&#x20;   ↓

CT Image Preparation

&#x20;   ↓

Train / Test Split

&#x20;   ↓

Real CT Images

&#x20;   ↓

GAN-based Synthetic Image Generation

&#x20;   ↓

Model Training

&#x20;   ↓

Model Evaluation

&#x20;   ↓

Model Comparison

&#x20;   ↓

Grad-CAM++ Explainability

````



\## Deep Learning Models



The project includes experiments with Transformer-based architectures.





\### Vision Transformers



\* Vision Transformer (ViT)

\* DeiT

\* Swin Transformer

\* Swin Transformer V2

\* LeViT

\* BEiT



The main experimental focus of the current pipeline is:



\* Swin Transformer / Swin V2



\## GAN-Based Data Augmentation



GAN-based synthetic image generation is used to investigate whether additional synthetic CT images can improve classification performance.



The GAN training code is located in:



```text

src/gan/

```



Current implementation:



```text

src/gan/train\_dcgan\_ct.py

```



Synthetic images are intended to be used as an augmentation source rather than as a replacement for real CT images.



\## Explainability



Grad-CAM-based explainability is used to visualize image regions that contribute to model predictions.



The explainability scripts are located in:



```text

src/explainability/

```



Current files include:



```text

gradcam\_attention.py

gradcam\_final.py

lung\_cancer\_pro\_gui\_gradcam.py

```



Example Grad-CAM output:



```text

results/gradcam\_result.png

```



\## Evaluation



Model performance is evaluated using:



\* Accuracy

\* Precision

\* Recall / Sensitivity

\* F1-score

\* ROC-AUC

\* Confusion Matrix

\* ROC Curve



Example evaluation outputs are stored in:



```text

results/

├── confusion\_matrix.png

├── roc\_curve.png

└── gradcam\_result.png

```



\## Repository Structure



```text

lung-cancer-classification/

│

├── results/

│   ├── confusion\_matrix.png

│   ├── gradcam\_result.png

│   └── roc\_curve.png

│

├── src/

│   │

│   ├── data/

│   │   ├── export\_ct\_images.py

│   │   ├── inspect\_hdf5.py

│   │   ├── prepare\_ct\_dataset.py

│   │   └── split\_dataset.py

│   │

│   ├── evaluation/

│   │   ├── compare\_models.py

│   │   └── final\_evaluation.py

│   │

│   ├── explainability/

│   │   ├── gradcam\_attention.py

│   │   ├── gradcam\_final.py

│   │   └── lung\_cancer\_pro\_gui\_gradcam.py

│   │

│   ├── gan/

│   │   └── train\_dcgan\_ct.py

│   │

│   └── training/

│       ├── train\_transformer.py

│       ├── train\_vit.py

│       └── train\_vit\_eval.py

│

├── .gitattributes

└── README.md

```



\## Dataset



The project uses a CT scan image dataset for lung cancer classification.



The dataset is processed and prepared before model training.



The original dataset and large generated image collections are not included in this repository.



\## Installation



Clone the repository:



```bash

git clone https://github.com/NH-Sarker-Galib/lung-cancer-classification.git

cd lung-cancer-classification

```



Create a virtual environment:



```bash

python -m venv venv

```



Activate the environment on Windows:



```bash

venv\\Scripts\\activate

```



Install the required dependencies:



```bash

pip install -r requirements.txt

```



> Note: `requirements.txt` will be added to the repository separately.



\## Running the Project



\### Dataset Preparation



```bash

python src/data/inspect\_hdf5.py

```



```bash

python src/data/prepare\_ct\_dataset.py

```



```bash

python src/data/split\_dataset.py

```



\### GAN Training



```bash

python src/gan/train\_dcgan\_ct.py

```



\### Model Training



```bash

python src/training/train\_vit.py

```



```bash

python src/training/train\_transformer.py

```



\### Model Evaluation



```bash

python src/evaluation/final\_evaluation.py

```



\### Model Comparison



```bash

python src/evaluation/compare\_models.py

```



\## Results



Example outputs generated during the experiments are available in:



```text

results/

```



These include:



\* Confusion matrix

\* ROC curve

\* Grad-CAM visualization



Quantitative results will be documented here as the experimental evaluation is finalized.



\## Large Files



Large datasets, generated image collections, and trained model checkpoints are intentionally excluded from the repository.



Examples of excluded files include:



```text

\*.h5

\*.hdf5

\*.pth

\*.pt

\*.ckpt

```



\## Disclaimer



This project is developed for academic and research purposes.



The predictions generated by this system are not intended to be used as a standalone medical diagnosis or as a replacement for evaluation by qualified medical professionals.



\## Author



\*\*Md. Naimul Haque Sarker Galib\*\*



Computer Science and Engineering

IUBAT — International University of Business Agriculture and Technology



GitHub: \[https://github.com/NH-Sarker-Galib](https://github.com/NH-Sarker-Galib)







