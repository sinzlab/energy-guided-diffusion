from lib.nnvision.nnvision.models.trained_models.v4_task_driven import task_driven_ensemble_1, task_driven_ensemble_2
from lib.nnvision.nnvision.models.trained_models.v4_data_driven import \
    v4_multihead_attention_ensemble_model, v4_multihead_attention_ensemble_model_2

task_driven_ensemble_1.eval()
task_driven_ensemble_2.eval()

v4_multihead_attention_ensemble_model.eval()
v4_multihead_attention_ensemble_model_2.eval()

models = {
    'task_driven': {
        'train': task_driven_ensemble_1,
        'val': task_driven_ensemble_2,
        'cross-val': v4_multihead_attention_ensemble_model,
    },
    'v4_multihead_attention': {
        'train': v4_multihead_attention_ensemble_model,
        'val': v4_multihead_attention_ensemble_model_2,
        'cross-val': task_driven_ensemble_1,
    },
    'cross': {
        'train': task_driven_ensemble_1,
        'val': task_driven_ensemble_2,
        'cross-val': v4_multihead_attention_ensemble_model,
    }
}