# DGTCIA
Enhanced Crested Ibis Algorithm: Performance Validation in Benchmark Functions, Engineering Problem, and Application in Brain Tumor Detection

## Highlights
• We propose a novel Dual Gene Targeting Crested Ibis Algorithm (DGTCIA) for global optimization.  
• We investigate the performance of DGTCIA in comprehensive numerical experiments.  
• We apply DGTCIA to ensemble learning and propose a hybrid deep learning approach DGTCIA-Ensemble for brain tumor detection.  

## Abstract
This paper presents a novel optimization method named Dual Gene Targeting Crested Ibis Algorithm (DGTCIA), an extension of the recently proposed Crested Ibis Algorithm (CIA). While CIA is effective in solving optimization problems, it suffers from limitations of metaheuristic algorithms (MAs), including premature convergence, early stagnation, and a narrow application range. To address these issues, we integrate a tailored dual gene targeting operator into CIA to form DGTCIA. We conducted comprehensive numerical experiments on IEEE-CEC2017 and CEC2022 to evaluate the performance of DGTCIA in functional optimization problems and six engineering problems and investigate the capacity of real-world applications. Thirteen state-of-the-art and well-known optimizers are employed as competitors. The experimental results and statistical analysis confirm the effectiveness and efficiency of our proposed DGTCIA. Furthermore, we apply DGTCIA to ensemble learning and propose DGTCIA-Ensemble for brain tumor detection, where several pre-trained deep learning models are fine-tuned in the brain tumor dataset, while the top three models with the highest accuracies are chosen for ensemble with the soft voting scheme. Each model is assigned a DGTCIA-optimized weight to balance the prediction. Experiments in the public dataset confirm that DGTCIA-Ensemble achieves the best precision with 98.635%, accuracy with 98.619%, recall with 98.619%, and F1 score with 98.621%. The source code of this research can be downloaded from https://github.com/RuiZhong961230/DGTCIA.

## Citation
@article{Zhong:25,  
title = {Enhanced Crested Ibis Algorithm: Performance Validation in Benchmark Functions, Engineering Problem, and Application in Brain Tumor Detection},  
journal = {Expert Systems with Applications},  
pages = {128231},  
year = {2025},  
issn = {0957-4174},  
doi = {https://doi.org/10.1016/j.eswa.2025.128231 },  
author = {Rui Zhong and Abdelazim G. Hussien and Essam H. Houssein and Jun Yu},  
}

## Datasets and Libraries
CEC benchmarks and Engineering problems are provided by opfunu==1.0.0 and enoppy==0.1.1 libraries, respectively. The brain tumor dataset is downloaded from Kaggle https://www.kaggle.com/datasets/rm1000/brain-tumor-mri-scans/data.

## Contact
If you have any questions, please don't hesitate to contact zhongrui[at]iic.hokudai.ac.jp.
