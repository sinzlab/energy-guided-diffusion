"""
This file contains the task-driven ensemble models for the gray-scale images.
"""
import torch


def get_task_driven_ensemble():
    from lib.nnvision.nnvision.models.trained_models.v4_task_driven import (
        task_driven_ensemble_1,
        task_driven_ensemble_2,
    )

    task_driven_ensemble_1.eval()
    task_driven_ensemble_2.eval()

    if torch.cuda.is_available():
        task_driven_ensemble_1.cuda()
        task_driven_ensemble_2.cuda()

    return task_driven_ensemble_1, task_driven_ensemble_2
