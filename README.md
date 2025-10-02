# Knowledge Distillation in Model Zoos – Variance and Compression Robustness

## 📌 Project Overview
This project investigates whether **Teacher–Student Knowledge Distillation (KD)** produces robust, smaller models with comparable performance to large networks. We focus on building **model zoos** of distilled and sparsified models to statistically assess **variance, robustness, and efficiency**.  

Our target application is **on-board satellite image classification**, where energy and memory resources are highly constrained.  

## 🚀 Objectives
- Train large **teacher models** on EuroSAT as baselines  
- Distill **smaller student models** guided by teachers  
- Build **model zoos** of students to analyze robustness across random seeds  
- Apply **sparsification** (magnitude pruning, variational dropout) for additional compression  
- Benchmark models on **accuracy, robustness (variance), and efficiency** (FLOPs, parameters, inference time)  

## 📊 Dataset
- **EuroSAT RGB** (3 bands, 64×64, 10 classes)  
- **EuroSAT Multispectral** (13 Sentinel-2 bands, 64×64, 10 classes)  
- ~27k images, balanced across classes  
- Augmentations: flips, rotations, resized crops, Gaussian noise, MixUp, CutMix  

## 🛠️ Approach
1. **Train Teacher Models** (over-parameterized CNNs/ResNets)  
2. **Distill Student Models** (e.g., ResNet-10) using cross-entropy + KL divergence loss  
3. **Model Zoos**: Train many student instances with different seeds → analyze robustness  
4. **Sparsification**: Apply pruning and variational dropout to distilled students  
5. **Evaluation**: Compare all models against teachers and non-distilled baselines  

## 📅 Roadmap
- [x] Train teacher models  
- [ ] Train student models (with & without KD)  
- [ ] Build distilled model zoo → analyze variance  
- [ ] Apply sparsification → second zoo  
- [ ] Compare against baselines  

## 🔍 Novelty
- Combines **distillation + sparsification** under a **robustness framework**  
- Uses **model zoos** to statistically analyze performance variance  

## 🌍 Impact
Reliable, efficient small models are crucial for:  
- On-board **satellite data processing**  
- **Edge AI applications** in resource-constrained environments  
- Geospatial monitoring, disaster response, climate observation  

## 📚 Key References
- Hinton et al. (2015) – Distilling the Knowledge in a Neural Network  
- Wenzel et al. (2020) – Model Zoo: Generalization & Robustness  
- Furlanello et al. (2018) – Born-Again Networks  
- Honegger et al. (2023) – EuroSAT Model Zoo  