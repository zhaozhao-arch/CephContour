# CephContour: A Multi-Center Dataset and Contour-Guided Multi-Task Cephalometric Diagnosis

Official PyTorch implementation and dataset repository for the paper "CephContour: A Multi-Center Dataset and Contour-Guided Multi-Task Cephalometric Diagnosis".

## Abstract

Cephalometric diagnosis plays a pivotal role in orthodontics. Currently, the dominant landmark-based ``detect-then-measure'' pipelines are inherently hindered by error accumulation. Although direct automated diagnostic classification offers a promising alternative to bypass this issue, its development is severely bottlenecked by the scarcity of suitable datasets. To address these limitations, we construct a multi-center dataset of 2,000 lateral cephalograms by integrating three public benchmarks with our newly collected CephContour-400 dataset. This dataset unifies six clinical diagnostic tasks based on American Board of Orthodontics (ABO) standards and provides expert-annotated cephalometric tracings for the CephContour-400 subset, enabling contour-aware learning and evaluation. Leveraging this dataset, we propose a Contour-Guided Multi-Task Framework. Specifically, we transform tracing-based structural contours into dense unsigned distance fields, which serve as geometric priors explicitly injects into multi-scale visual representations. Extensive experiments under a cross-center evaluation protocol demonstrate that our framework establishes a new state-of-the-art, achieving 93.51\% average accuracy and 98.75\% AUC, providing a robust and clinically reliable solution for cephalometric diagnosis. 

## Highlights

* Multi-Center Dataset: Integrates 2,000 lateral cephalograms across multiple centers, standardizing six clinical diagnostic tasks based on ABO standards.
* Dense Structural Tracings: Provides expert-annotated cephalometric tracings for the CephContour-400 subset, bypassing traditional error-prone sparse landmarks.
* Novel Architecture: Introduces the Hierarchical Geometric Injection Module (HGIM) to rectify visual features via contour-derived dense unsigned distance fields (UDFs). 

