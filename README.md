# Découverte de Molécules pour Nouveaux Médicaments via l'IA Générative Hybride

**Auteurs** : Chihab Ouchen & Ahmed Ouidani
**Institution** : Faculté Polydisciplinaire de Safi, Université Cadi Ayyad
**Année** : 2026

## 📜 Résumé du Projet

Ce projet explore l'utilisation de modèles génératifs profonds (Deep Generative Models) pour accélérer la phase de découverte de médicaments (Drug Discovery). Nous nous concentrons sur la génération de graphes moléculaires valides et originaux en hybridant plusieurs paradigmes architecturaux : Auto-encodeurs Variationnels (VAE), Réseaux Antagonistes Génératifs (GAN) et Modèles de Diffusion (Diffusion Models).

L'objectif est de générer des molécules qui satisfont les contraintes de valence chimique tout en maximisant des propriétés cibles comme le QED (Quantitative Estimation of Drug-likeness) et le LogP.

## 🏗️ Architectures Implémentées

Le dépôt contient le code source de trois variantes architecturales distinctes décrites dans notre rapport de recherche :

### 1. Graph GAN-VAE (`graph_gan_vae/`)
Une fusion stratégique qui utilise :
*   **VAE** : Pour structurer un espace latent continu et régulier.
*   **GAN** : Pour générer des structures réalistes (notamment les cycles aromatiques) en échantillonnant depuis cet espace latent.
*   *Performance* : 100% de validité et une excellente distribution des propriétés physico-chimiques.

### 2. C-GLD: Conditional Graph Latent Diffusion (`C-GLD/`)
Une approche basée sur les modèles de diffusion latents :
*   L'entraînement se fait dans l'espace compressé d'un Auto-encodeur.
*   Conditionnement explicite par les propriétés désirées (QED, Solubilité).

### 3. Ultimate Gen: Graph Transformer + Diffusion (`ultimate_gen/`)
Notre modèle le plus avancé combinant :
*   **Graph Transformers** : Pour capturer les dépendances à longue portée entre atomes distants via des mécanismes d'attention.
*   **Discrete Diffusion** : Un processus de débruitage itératif pour construire le graphe atome par atome et liaison par liaison.
*   *Performance* : 100% de nouveauté et capacité à générer des structures complexes.

## 📂 Structure du Dépôt

```
.
├── assets/                 # Images et visualisations pour le rapport (Générées par les modèles)
├── graph_gan_vae/          # Code source du modèle GAN-VAE
├── ultimate_gen/           # Code source du modèle Transformer-Diffusion
├── C-GLD/                  # Code source du modèle Latent Diffusion
├── report.tex              # Rapport scientifique complet (Format LaTeX)
├── requirements.txt        # Dépendances Python
└── README.md               # Ce fichier
```

## 🚀 Installation et Utilisation

### Pré-requis
*   Python 3.8+
*   PyTorch (avec support CUDA recommandé)
*   PyTorch Geometric
*   RDKit

### Installation
```bash
git clone https://github.com/chihab999/Gen_Ai_project.git
cd drug-discovery-genai
pip install -r requirements.txt
```

### Lancer une évaluation
Pour générer des molécules avec le modèle Graph GAN-VAE par exemple :

```bash
cd graph_gan_vae
python evaluate_advanced.py
```
Les résultats (images des molécules, distributions) seront sauvegardés dans le dossier `evaluation_results/`.

## 📊 Résultats Clés

Nous avons évalué nos modèles sur le dataset QM9.

| Métrique | Graph GAN-VAE | Ultimate Gen |
|----------|---------------|--------------|
| **Validité** | **100%** | **100%** |
| **Unicité** | 98% | **100%** |
| **Nouveauté** | **100%** | **100%** |

*(Voir le rapport complet `report.pdf` généré depuis `report.tex` pour l'analyse détaillée)*

## 👥 Équipe

Ce travail a été réalisé dans le cadre du Master Data Science et IA.
*   **Chihab Ouchen** (chihabouchen11@gmail.com)
*   **Ahmed Ouidani** (A.ouidani9533@uca.ac.ma)

---
*Faculté Polydisciplinaire de Safi - 2026*
