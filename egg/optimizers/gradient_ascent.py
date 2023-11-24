from typing import Callable, Dict, Tuple, Type, Optional

import torch
from mei import optimization
from mei.import_helpers import import_object
from mei.tracking import Tracker
from torch import Tensor, nn
from torch.nn import Module


class SingleUnitModel(nn.Module):
    def __init__(self, model, unit_idx, data_key):
        super().__init__()
        self.model = model
        self.unit_idx = unit_idx
        self.data_key = data_key

    def forward(self, *args, **kwargs):
        return self.model(*args, data_key=self.data_key, **kwargs)[..., self.unit_idx]


def gradient_ascent(
    model: Module,
    config: Dict,
    data_key: str,
    seed: int,
    shape: Tuple[int, ...] = (1, 1, 100, 100),
    unit: Optional[int] = None,
    set_seed: Callable = torch.manual_seed,
    mei_class: Type = optimization.MEI,
    import_func: Callable = import_object,
    optimize_func: Callable = optimization.optimize,
    tracker_cls: Type[Tracker] = Tracker,
) -> Tuple[Tensor, float, Dict]:
    """Generates a MEI using gradient ascent.

    The value corresponding to the "device" key must be either "cpu" or "cuda". The "transform",
    "regularization", "precondition" and "postprocessing" components are optional and can be omitted. All "kwargs" items
    in the config are optional and can be omitted as well. Furthermore the "objectives" item is optional and can be
    omitted. Example config:

        {
            "device": "cuda",
            "initial": {
                "path": "path.to.initial",
                "kwargs": {"initial_kwarg1": 0, "initial_kwarg2": 1},
            },
            "optimizer": {
                "path": "path.to.optimizer",
                "kwargs": {"optimizer_kwarg1": 0, "optimizer_kwarg2": 1},
            },
            "stopper": {
                "path": "path.to.stopper",
                "kwargs": {"stopper_kwarg1": 0, "stopper_kwarg2": 0},
            },
            "transform": {
                "path": "path.to.transform",
                "kwargs": {"transform_kwarg1": 0, "transform_kwarg2": 1},
            },
            "regularization": {
                "path": "path.to.regularization",
                "kwargs": {"regularization_kwarg1": 0, "regularization_kwarg2": 1},
            },
            "precondition": {
                "path": "path.to.precondition",
                "kwargs": {"precondition_kwarg1": 0, "precondition_kwarg2": 1},
            },
            "postprocessing": {
                "path": "path.to.postprocessing",
                "kwargs": {"postprocessing_kwarg1": 0, "postprocessing_kwarg2": 1},
            },
            "objectives": [
                {"path": "path.to.objective1", "kwargs": {"objective1_kwarg1": 0, "objective1_kwarg2": 1}},
                {"path": "path.to.objective2", "kwargs": {"objective2_kwarg1": 0, "objective2_kwarg2": 1}},
            ],
        }

    Args:
        dataloaders: NNFabrik-style dataloader dictionary.
        model: Callable object that will receive a tensor and must return a tensor containing a single float.
        config: Configuration dictionary. See above for an explanation and example.
        seed: Integer used to make the MEI generation process reproducible.
        set_seed: For testing purposes.
        get_dims: For testing purposes.
        mei_class: For testing purposes.
        import_func: For testing purposes.
        optimize_func: For testing purposes.
        tracker_cls: For testing purposes.

    Returns:
        The MEI, the final evaluation as a single float and the log of the tracker.
    """
    if unit is not None:
        model = SingleUnitModel(model, unit, data_key)

    for component_name, component_config in config.items():
        if component_name in (
            "device",
            "objectives",
            "n_meis",
            "mei_shape",
            "model_forward_kwargs",
            "transparency",
            "transparency_weight",
            "inhibitory",
        ):
            continue
        if "kwargs" not in component_config:
            component_config["kwargs"] = dict()

    if "objectives" not in config:
        config["objectives"] = []
    else:
        for obj in config["objectives"]:
            if "kwargs" not in obj:
                obj["kwargs"] = dict()

    set_seed(seed)
    model.eval()
    model.to(config["device"])

    n_meis = config.get("n_meis", 1)
    # model_forward_kwargs = config.get("model_forward_kwargs", dict())
    # model.forward_kwargs.update(model_forward_kwargs)

    create_initial_guess = import_func(
        config["initial"]["path"], config["initial"]["kwargs"]
    )
    initial_guess = create_initial_guess(n_meis, *shape[1:]).to(
        config["device"]
    )  # (1*1*h*w)

    transparency = config.get("transparency", None)
    if transparency:
        initial_alpha = (torch.ones(n_meis, 1, *shape[2:]) * 0.5).to(config["device"])
        # add transparency by concatenate alpha channel
        initial_guess = torch.cat((initial_guess, initial_alpha), dim=1)
    transparency_weight = config.get("transparency_weight", 1.0)
    inhibitory = config.get("inhibitory", None)

    optimizer = import_func(
        config["optimizer"]["path"],
        dict(params=[initial_guess], **config["optimizer"]["kwargs"]),
    )
    stopper = import_func(config["stopper"]["path"], config["stopper"]["kwargs"])

    objectives = {
        o["path"]: import_func(o["path"], o["kwargs"]) for o in config["objectives"]
    }
    tracker = tracker_cls(**objectives)

    optional_names = (
        "transform",
        "regularization",
        "precondition",
        "postprocessing",
        "background",
    )
    optional = {
        n: import_func(config[n]["path"], config[n]["kwargs"])
        for n in optional_names
        if n in config
    }
    mei = mei_class(
        model,
        initial=initial_guess,
        optimizer=optimizer,
        transparency=transparency,
        inhibitory=inhibitory,
        transparency_weight=transparency_weight,
        **optional,
    )

    final_evaluation, mei = optimize_func(mei, stopper, tracker)
    return mei, final_evaluation, tracker.log