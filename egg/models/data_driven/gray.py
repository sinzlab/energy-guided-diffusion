"""
This file contains the trained models for the data-driven approach.
"""
import torch


def get_v4_multihead_attention_ensemble_model():
    from lib.nnvision.nnvision.models.trained_models.v4_data_driven import (
        v4_multihead_attention_ensemble_model,
        v4_multihead_attention_ensemble_model_2,
    )

    v4_multihead_attention_ensemble_model.eval()
    v4_multihead_attention_ensemble_model_2.eval()

    if torch.cuda.is_available():
        v4_multihead_attention_ensemble_model.cuda()
        v4_multihead_attention_ensemble_model_2.cuda()

    return v4_multihead_attention_ensemble_model, v4_multihead_attention_ensemble_model_2

