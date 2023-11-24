from egg.models.data_driven.gray import get_v4_multihead_attention_ensemble_model
from egg.models.task_driven.gray import get_task_driven_ensemble
from egg.models.task_driven.color import (
    get_color_task_driven_ensemble,
    get_colorgray_task_driven_ensemble,
    get_gray_task_driven_ensemble
)

import torch.nn as nn


# lazy dict that only loads the model when it is called
# this is to avoid loading all models at once
class LazyModelDict:
    def __init__(self):
        super().__init__()

        self.tasks = [
            "task_driven",
            "v4_multihead_attention",
            "cross",
            "color",
            "colorgray",
            "gray",
        ]

        self.models = {}

    def __getitem__(self, key):
        if key in self.tasks and not key in self.models:
            self.models[key] = self._load_model(key)

        return self.models[key]

    def _load_model(self, key):
        if key == "task_driven":
            task_driven_ensemble_1, task_driven_ensemble_2 = get_task_driven_ensemble()
            v4_multihead_attention_ensemble_model, v4_multihead_attention_ensemble_model_2 = get_v4_multihead_attention_ensemble_model()

            return {
                "train": task_driven_ensemble_1,
                "val": task_driven_ensemble_2,
                "cross-val": v4_multihead_attention_ensemble_model,
            }

        if key == "v4_multihead_attention":
            task_driven_ensemble_1, task_driven_ensemble_2 = get_task_driven_ensemble()
            v4_multihead_attention_ensemble_model, v4_multihead_attention_ensemble_model_2 = get_v4_multihead_attention_ensemble_model()

            return {
                "train": nn.DataParallel(v4_multihead_attention_ensemble_model),
                "val": nn.DataParallel(v4_multihead_attention_ensemble_model_2),
                "cross-val": nn.DataParallel(task_driven_ensemble_1),
            }

        if key == "cross":
            task_driven_ensemble_1, task_driven_ensemble_2 = get_task_driven_ensemble()
            v4_multihead_attention_ensemble_model, v4_multihead_attention_ensemble_model_2 = get_v4_multihead_attention_ensemble_model()

            return {
                "train": task_driven_ensemble_1,
                "val": task_driven_ensemble_2,
                "cross-val": v4_multihead_attention_ensemble_model,
            }

        if key == "color":
            color_task_driven_ensemble = get_color_task_driven_ensemble()
            return {
                "train": color_task_driven_ensemble,
                "val": color_task_driven_ensemble,
                "cross-val": color_task_driven_ensemble,
            }

        if key == "colorgray":
            colorgray_task_driven_ensemble = get_colorgray_task_driven_ensemble()
            return {
                "train": colorgray_task_driven_ensemble,
                "val": colorgray_task_driven_ensemble,
                "cross-val": colorgray_task_driven_ensemble,
            }

        if key == "gray":
            gray_task_driven_ensemble = get_gray_task_driven_ensemble()
            return {
                "train": gray_task_driven_ensemble,
                "val": gray_task_driven_ensemble,
                "cross-val": gray_task_driven_ensemble,
            }


models = LazyModelDict()
