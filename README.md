
# Simple GPT-like Model with Torch

## Overview

This repository demonstrates how to create a simple GPT-like model (decoder-only) from scratch using PyTorch. The project processes **Georgian text** from *"ვეფხისტყაოსანი"* ("The Knight in the Panther's Skin") by Shota Rustaveli as a toy dataset.

---

## Purpose

The primary goal is to:
- Learn and experiment with PyTorch by building a basic GPT-like model.
- Implement a decoder with multi-head attention.
- Use Georgian text for tokenization, model training, and evaluation.

---

## How to Use

All the code for this project is implemented and executed in the **`playground.ipynb`** notebook. You can replicate all steps, from data loading to model training and evaluation, by running the cells in the notebook.

1. **Install Dependencies**  
   Ensure Python and PyTorch are installed in your environment.

2. **Run the Notebook**  
   Open `playground.ipynb` in your preferred Jupyter environment:
   ```bash
   jupyter notebook playground.ipynb
   ```
   Follow the step-by-step instructions in the notebook to train and test the model.

---

## Repository Contents

```
📦 Repository
 ├── bigram_model_state.pth             # Weights for the bigram model.
 ├── character_tokenizer.pkl            # Character-level tokenizer.
 ├── config.py                          # Configuration for model hyperparameters.
 ├── custom_data_loader.py              # Data loader logic.
 ├── custom_models.py                   # Custom decoder model implementation.
 ├── custom_tokenizer.py                # Tokenizer for Georgian text.
 ├── decoder_model_multihead_blocks.pth # Trained decoder weights.
 ├── playground.ipynb                   # Jupyter notebook for running all code.
 ├── trainer.py                         # Training utilities (imported into the notebook).
 ├── vefxistyaosani.txt                 # Georgian text dataset.
```

---

## Key Features

- **All-in-One Notebook**: All steps (data preparation, training, and evaluation) are integrated into `playground.ipynb`.
- **Decoder Architecture**: Implements a basic decoder-only model with multi-head attention.
- **Character-Level Tokenization**: Works directly with Georgian characters for simplicity.
- **Lightweight and Simple**: No external setup; everything runs within the notebook.

This project is designed for learning and experimentation. Feel free to tweak and modify the notebook to suit your goals!

---
## License

This project is licensed under the MIT License - see the `LICENSE` file for details.
