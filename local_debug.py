import collections
import copy
import inspect
import os
from collections import OrderedDict
from collections.abc import Iterable
from dataclasses import dataclass, fields, is_dataclass
from functools import partial
from pathlib import Path
from typing import Any, Optional, Union
import re
import torch
from packaging import version
from PIL import Image
from safetensors import safe_open
from torch import nn

from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.utils.generic import torch_int, is_tensor
import enum
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class BackboneType(enum.Enum):
    TRANSFORMERS = "transformers"


def is_torch_available():
    return True


def get_torch_version():
    return "2.2"


def model_type_to_module_name(key) -> str:
    return key


default_home = os.path.join(os.path.expanduser("~"), ".cache")
HF_HOME = os.path.expandvars(
    os.path.expanduser(
        os.getenv(
            "HF_HOME",
            os.path.join(os.getenv("XDG_CACHE_HOME", default_home), "huggingface"),
        )
    )
)

default_cache_path = os.path.join(HF_HOME, "hub")
default_assets_cache_path = os.path.join(HF_HOME, "assets")

# Legacy env variables
HUGGINGFACE_HUB_CACHE = os.getenv("HUGGINGFACE_HUB_CACHE", default_cache_path)

HF_HUB_CACHE = os.path.expandvars(
    os.path.expanduser(
        os.getenv(
            "HF_HUB_CACHE",
            HUGGINGFACE_HUB_CACHE,
        )
    )
)


def try_to_load_from_cache(
    repo_id: str,
    filename: str,
    cache_dir: Union[str, Path, None] = None,
    revision: Optional[str] = None,
    repo_type: Optional[str] = None,
):
    if revision is None:
        revision = "main"
    if repo_type is None:
        repo_type = "model"
    if cache_dir is None:
        cache_dir = HF_HUB_CACHE

    object_id = repo_id.replace("/", "--")
    repo_cache = os.path.join(cache_dir, f"{repo_type}s--{object_id}")
    if not os.path.isdir(repo_cache):
        # No cache for this model
        return None

    refs_dir = os.path.join(repo_cache, "refs")
    snapshots_dir = os.path.join(repo_cache, "snapshots")

    # Resolve refs (for instance to convert main to the associated commit sha)
    if os.path.isdir(refs_dir):
        revision_file = os.path.join(refs_dir, revision)
        if os.path.isfile(revision_file):
            with open(revision_file) as f:
                revision = f.read()

    # Check if revision folder exists
    if not os.path.exists(snapshots_dir):
        return None
    cached_shas = os.listdir(snapshots_dir)
    if revision not in cached_shas:
        # No cache for this revision and we won't try to return a random revision
        return None

    # Check if file exists in cache
    cached_file = os.path.join(snapshots_dir, revision, filename)
    return cached_file if os.path.isfile(cached_file) else None


def _align_output_features_output_indices(
    out_features: Optional[list[str]],
    out_indices: Optional[Union[list[int], tuple[int]]],
    stage_names: list[str],
):
    if out_indices is None and out_features is None:
        out_indices = [len(stage_names) - 1]
        out_features = [stage_names[-1]]
    elif out_indices is None and out_features is not None:
        out_indices = [stage_names.index(layer) for layer in out_features]
    elif out_features is None and out_indices is not None:
        out_features = [stage_names[idx] for idx in out_indices]
    return out_features, out_indices


def verify_out_features_out_indices(
    out_features: Optional[Iterable[str]],
    out_indices: Optional[Iterable[int]],
    stage_names: Optional[Iterable[str]],
):
    if stage_names is None:
        raise ValueError("Stage_names must be set for transformers backbones")

    if out_features is not None:
        if not isinstance(out_features, (list,)):
            raise ValueError(f"out_features must be a list got {type(out_features)}")
        if any(feat not in stage_names for feat in out_features):
            raise ValueError(
                f"out_features must be a subset of stage_names: {stage_names} got {out_features}"
            )
        if len(out_features) != len(set(out_features)):
            raise ValueError(
                f"out_features must not contain any duplicates, got {out_features}"
            )
        if out_features != (
            sorted_feats := [feat for feat in stage_names if feat in out_features]
        ):
            raise ValueError(
                f"out_features must be in the same order as stage_names, expected {sorted_feats} got {out_features}"
            )

    if out_indices is not None:
        if not isinstance(out_indices, list):
            raise ValueError(f"out_indices must be a list, got {type(out_indices)}")
        # Convert negative indices to their positive equivalent: [-1,] -> [len(stage_names) - 1,]
        positive_indices = tuple(
            idx % len(stage_names) if idx < 0 else idx for idx in out_indices
        )
        if any(idx for idx in positive_indices if idx not in range(len(stage_names))):
            raise ValueError(
                f"out_indices must be valid indices for stage_names {stage_names}, got {out_indices}"
            )
        if len(positive_indices) != len(set(positive_indices)):
            msg = f"out_indices must not contain any duplicates, got {out_indices}"
            msg += (
                f"(equivalent to {positive_indices}))"
                if positive_indices != out_indices
                else ""
            )
            raise ValueError(msg)
        if positive_indices != tuple(sorted(positive_indices)):
            sorted_negative = [
                idx
                for _, idx in sorted(
                    zip(positive_indices, out_indices), key=lambda x: x[0]
                )
            ]
            raise ValueError(
                f"out_indices must be in the same order as stage_names, expected {sorted_negative} got {out_indices}"
            )

    if out_features is not None and out_indices is not None:
        if len(out_features) != len(out_indices):
            raise ValueError(
                "out_features and out_indices should have the same length if both are set"
            )
        if out_features != [stage_names[idx] for idx in out_indices]:
            raise ValueError(
                "out_features and out_indices should correspond to the same stages if both are set"
            )


def get_aligned_output_features_output_indices(
    out_features: Optional[list[str]],
    out_indices: Optional[Union[list[int], tuple[int]]],
    stage_names: list[str],
) -> tuple[list[str], list[int]]:
    out_indices = list(out_indices) if out_indices is not None else None
    # First verify that the out_features and out_indices are valid
    verify_out_features_out_indices(
        out_features=out_features, out_indices=out_indices, stage_names=stage_names
    )
    output_features, output_indices = _align_output_features_output_indices(
        out_features=out_features, out_indices=out_indices, stage_names=stage_names
    )
    # Verify that the aligned out_features and out_indices are valid
    verify_out_features_out_indices(
        out_features=output_features,
        out_indices=output_indices,
        stage_names=stage_names,
    )
    return output_features, output_indices


def _find_mismatched_keys(
    model: "LocalPretrainedModel",
    state_dict: Optional[dict],
    checkpoint_files: Optional[list[str]],
    ignore_mismatched_sizes: bool,
    keys_to_rename_mapping: dict[str, str],
    weights_only: bool,
) -> tuple[list[str], list[tuple[int, int]]]:
    # An error will be raised later on anyway if there is a mismatch - this avoids running the rest of this function
    # if there are no mismatch (which is almost always the case)
    if not ignore_mismatched_sizes:
        return [], []

    if state_dict is not None:
        checkpoint_files = [""]

    model_state_dict = model.state_dict()
    mismatched_keys = []
    mismatched_shapes = []
    for shard_file in checkpoint_files:
        # If shard_file is "", we use the existing state_dict instead of loading it
        if shard_file != "":
            state_dict = load_state_dict(
                shard_file,
                map_location="meta",
                weights_only=weights_only,
            )

        # Fix the key names
        new_state_dict = {
            keys_to_rename_mapping[k]: v
            for k, v in state_dict.items()
            if k in keys_to_rename_mapping
        }

        for key, tensor in new_state_dict.items():
            if key in model_state_dict and tensor.shape != model_state_dict[key].shape:
                # This skips size mismatches for 4-bit weights. Two 4-bit values share an 8-bit container, causing size differences.
                # Without matching with module type or parameter type it seems like a practical way to detect valid 4bit weights.
                if not (
                    tensor.shape[-1] == 1
                    and tensor.numel() * 2 == model_state_dict[key].numel()
                ):
                    mismatched_keys.append(key)
                    mismatched_shapes.append(
                        (tensor.shape, model_state_dict[key].shape)
                    )

    return mismatched_keys, mismatched_shapes


str_to_torch_dtype = {
    "BOOL": torch.bool,
    "U8": torch.uint8,
    "I8": torch.int8,
    "I16": torch.int16,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "I32": torch.int32,
    "F32": torch.float32,
    "F64": torch.float64,
    "I64": torch.int64,
    "F8_E4M3": torch.float8_e4m3fn,
    "F8_E5M2": torch.float8_e5m2,
    "U16": torch.uint16,
    "U32": torch.uint32,
    "U64": torch.uint64,
}


def load_state_dict(
    checkpoint_file: Union[str, os.PathLike],
    map_location: Optional[Union[str, torch.device]] = "cpu",
    weights_only: bool = True,
):
    if checkpoint_file.endswith(".safetensors"):
        with safe_open(checkpoint_file, framework="pt") as f:
            metadata = f.metadata()

            if metadata is not None and metadata.get("format") not in [
                "pt",
            ]:
                raise OSError(
                    f"The safetensors archive passed at {checkpoint_file} does not contain the valid metadata. Make sure "
                    "you save your model with the `save_pretrained` method."
                )
            state_dict = {}
            for k in f.keys():
                if map_location == "meta":
                    _slice = f.get_slice(k)
                    k_dtype = _slice.get_dtype()
                    if k_dtype in str_to_torch_dtype:
                        dtype = str_to_torch_dtype[k_dtype]
                    else:
                        raise ValueError(
                            f"Cannot load safetensors of unknown dtype {k_dtype}"
                        )
                    state_dict[k] = torch.empty(
                        size=_slice.get_shape(), dtype=dtype, device="meta"
                    )
                else:
                    state_dict[k] = f.get_tensor(k)
            return state_dict
    else:
        raise OSError(f"{checkpoint_file} is not a .safetensors file.")


def is_local_dist_rank_0():
    return (
        torch.distributed.is_available()
        and torch.distributed.is_initialized()
        and int(os.environ.get("LOCAL_RANK", "-1")) == 0
    )


def strtobool(val):
    val = val.lower()
    if val in {"y", "yes", "t", "true", "on", "1"}:
        return 1
    if val in {"n", "no", "f", "false", "off", "0"}:
        return 0
    raise ValueError(f"invalid truth value {val!r}")


def is_fsdp_enabled():
    return (
        torch.distributed.is_available()
        and torch.distributed.is_initialized()
        and strtobool(os.environ.get("ACCELERATE_USE_FSDP", "False")) == 1
        and strtobool(os.environ.get("FSDP_CPU_RAM_EFFICIENT_LOADING", "False")) == 1
    )


def _infer_parameter_dtype(
    model: "LocalPretrainedModel",
    param_name: str,
) -> Union[bool, Optional[torch.dtype]]:
    old_param = model.get_parameter_or_buffer(param_name)
    casting_dtype = old_param.dtype
    return old_param is not None and old_param.is_contiguous(), casting_dtype


def get_module_from_name(module, tensor_name: str) -> tuple[Any, str]:
    if "." in tensor_name:
        module_name, tensor_name = tensor_name.rsplit(".", 1)
        module = module.get_submodule(module_name)
    return module, tensor_name


def _load_parameter_into_model(
    model: "LocalPretrainedModel", param_name: str, tensor: torch.Tensor
):
    module, param_type = get_module_from_name(model, param_name)
    # This will check potential shape mismatch if skipped before
    module.load_state_dict({param_type: tensor}, strict=False, assign=True)


@torch.no_grad()
def _load_state_dict_into_meta_model(
    model: "LocalPretrainedModel",
    state_dict: dict,
    shard_file: str,
    expected_keys: list[str],
    reverse_renaming_mapping: dict[str, str],
    device_map: Optional[dict] = None,
) -> tuple[Optional[dict], Optional[dict]]:
    tensor_device = "cpu"
    if device_map is not None and device_map.get("", None) is not None:
        if device_map[""] not in ("cpu", torch.device("cpu")):
            tensor_device = (
                device_map[""].index
                if isinstance(device_map[""], torch.device)
                else device_map[""]
            )

    is_meta_state_dict = shard_file.endswith(".safetensors")
    file_pointer = None
    if is_meta_state_dict:
        file_pointer = safe_open(shard_file, framework="pt", device=tensor_device)

    for param_name, empty_param in state_dict.items():
        if (
            param_name not in expected_keys
        ):  # when loading from ckpt, we skip param if doesnt exist in modeling
            continue
        # we need to use serialized_param_name as file pointer is untouched
        if is_meta_state_dict:
            # This is the name of the parameter as it appears on disk file
            serialized_param_name = reverse_renaming_mapping[param_name]
            param = file_pointer.get_slice(serialized_param_name)
        else:
            param = empty_param.to(tensor_device)  # It is actually not empty!
        to_contiguous, casting_dtype = _infer_parameter_dtype(
            model,
            param_name,
        )
        param = param[...]
        if casting_dtype is not None:
            param = param.to(casting_dtype)
        if to_contiguous:
            param = param.contiguous()
        if device_map is None:
            param_device = "cpu"

        if is_fsdp_enabled():
            param_device = "cpu" if is_local_dist_rank_0() else "meta"

        _load_parameter_into_model(model, param_name, param.to(param_device))

    if file_pointer is not None:
        file_pointer.__exit__(None, None, None)

    return None, None


def load_shard_file(args):
    (shard_file, state_dict, device_map, key_renaming_mapping, model_to_load) = args
    map_location = "cpu"
    if shard_file.endswith(".safetensors"):
        map_location = "meta"
    else:
        map_location = torch.device(
            [d for d in device_map.values() if d not in ["disk"]][0]
        )

    # If shard_file is "", we use the existing state_dict instead of loading it
    if shard_file != "":
        state_dict = load_state_dict(shard_file, map_location=map_location)

    # Fix the key names
    state_dict = {
        key_renaming_mapping[k]: v
        for k, v in state_dict.items()
        if k in key_renaming_mapping
    }

    error_msgs = []

    expected_keys = list(model_to_load.state_dict().keys())
    reverse_key_renaming_mapping = {v: k for k, v in key_renaming_mapping.items()}

    disk_offload_index, cpu_offload_index = _load_state_dict_into_meta_model(
        model_to_load,
        state_dict,
        shard_file,
        expected_keys,
        reverse_key_renaming_mapping,
        device_map=device_map,
    )

    return error_msgs, disk_offload_index, cpu_offload_index


class LocalPretrainedConfig:
    sub_configs: dict[str, type["LocalPretrainedConfig"]] = {}
    attribute_map: dict[str, str] = {}

    def __setattr__(self, key, value):
        if key in super().__getattribute__("attribute_map"):
            key = super().__getattribute__("attribute_map")[key]
        super().__setattr__(key, value)

    def __getattribute__(self, key):
        if key != "attribute_map" and key in super().__getattribute__("attribute_map"):
            key = super().__getattribute__("attribute_map")[key]
        return super().__getattribute__(key)

    def __init__(
        self,
        *,
        # All models common arguments
        output_hidden_states: bool = False,
        output_attentions: bool = False,
        return_dict: bool = True,
        torchscript: bool = False,
        torch_dtype: Optional[Union[str, "torch.dtype"]] = None,
        # Common arguments
        pruned_heads: Optional[dict[int, list[int]]] = None,
        tie_word_embeddings: bool = True,
        chunk_size_feed_forward: int = 0,
        is_encoder_decoder: bool = False,
        is_decoder: bool = False,
        cross_attention_hidden_size: Optional[int] = None,
        add_cross_attention: bool = False,
        tie_encoder_decoder: bool = False,
        # Fine-tuning task arguments
        architectures: Optional[list[str]] = None,
        finetuning_task: Optional[str] = None,
        id2label: Optional[dict[int, str]] = None,
        label2id: Optional[dict[str, int]] = None,
        num_labels: Optional[int] = None,
        task_specific_params: Optional[dict[str, Any]] = None,
        problem_type: Optional[str] = None,
        # Tokenizer kwargs
        tokenizer_class: Optional[str] = None,
        prefix: Optional[str] = None,
        bos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        sep_token_id: Optional[int] = None,
        decoder_start_token_id: Optional[int] = None,
        **kwargs,
    ):
        # Validation for some arguments
        if label2id is not None and not isinstance(label2id, dict):
            raise ValueError("Argument label2id should be a dictionary.")
        if id2label is not None and not isinstance(id2label, dict):
            raise ValueError("Argument id2label should be a dictionary.")
        if (
            num_labels is not None
            and id2label is not None
            and len(id2label) != num_labels
        ):
            logger.warning(
                f"You passed `num_labels={num_labels}` which is incompatible to "
                f"the `id2label` map of length `{len(id2label)}`."
            )
        if problem_type is not None and problem_type not in (
            "regression",
            "single_label_classification",
            "multi_label_classification",
        ):
            raise ValueError(
                f"The config parameter `problem_type` was not understood: received {problem_type} "
                "but only 'regression', 'single_label_classification' and 'multi_label_classification' are valid."
            )
        if (
            torch_dtype is not None
            and isinstance(torch_dtype, str)
            and is_torch_available()
        ):
            # we will start using self.torch_dtype in v5, but to be consistent with
            # from_pretrained's torch_dtype arg convert it to an actual torch.dtype object
            import torch

            torch_dtype = getattr(torch, torch_dtype)

        # Attributes common for all models
        self.return_dict = return_dict
        self.output_hidden_states = output_hidden_states
        self.torchscript = torchscript
        self.torch_dtype = torch_dtype
        self._output_attentions = output_attentions  # has public property

        # Less common kwargs, only used by some models
        self.pruned_heads = pruned_heads if pruned_heads is not None else {}
        self.tie_word_embeddings = tie_word_embeddings
        self.chunk_size_feed_forward = chunk_size_feed_forward

        # Encoder-decoder models attributes
        self.is_encoder_decoder = is_encoder_decoder
        self.is_decoder = is_decoder  # used in encoder-decoder models to differentiate encoder from decoder
        self.cross_attention_hidden_size = cross_attention_hidden_size
        self.add_cross_attention = add_cross_attention
        self.tie_encoder_decoder = tie_encoder_decoder

        # Fine-tuning task attributes
        self.architectures = architectures
        self.finetuning_task = finetuning_task
        self.id2label = id2label
        self.label2id = label2id
        self.task_specific_params = task_specific_params
        self.problem_type = problem_type

        # Tokenizer attributes
        self.tokenizer_class = tokenizer_class
        self.prefix = prefix
        self.bos_token_id = bos_token_id
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.sep_token_id = sep_token_id
        self.decoder_start_token_id = decoder_start_token_id

        # Name or path to the pretrained checkpoint
        self._name_or_path = str(kwargs.pop("name_or_path", ""))
        self._commit_hash = kwargs.pop("_commit_hash", None)

        # Attention implementation to use, if relevant (it sets it recursively on sub-configs)
        self._attn_implementation = kwargs.pop("attn_implementation", None)

        # Drop the transformers version info
        self.transformers_version = kwargs.pop("transformers_version", None)

        # Deal with gradient checkpointing
        if kwargs.get("gradient_checkpointing", False):
            warnings.warn(
                "Passing `gradient_checkpointing` to a config initialization is deprecated and will be removed in v5 "
                "Transformers. Using `model.gradient_checkpointing_enable()` instead, or if you are using the "
                "`Trainer` API, pass `gradient_checkpointing=True` in your `TrainingArguments`."
            )

        # Additional attributes without default values
        for key, value in kwargs.items():
            try:
                setattr(self, key, value)
            except AttributeError as err:
                logger.error(f"Can't set {key} with value {value} for {self}")
                raise err

        # TODO: remove later, deprecated arguments for TF models
        self.tf_legacy_loss = kwargs.pop("tf_legacy_loss", False)
        self.use_bfloat16 = kwargs.pop("use_bfloat16", False)

    @property
    def name_or_path(self) -> Optional[str]:
        return getattr(self, "_name_or_path", None)

    @name_or_path.setter
    def name_or_path(self, value):
        self._name_or_path = str(
            value
        )  # Make sure that name_or_path is a string (for JSON encoding)

    @classmethod
    def get_config_dict(
        cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        original_kwargs = copy.deepcopy(kwargs)
        # Get config dict associated with the base config file
        config_dict, kwargs = cls._get_config_dict(
            pretrained_model_name_or_path, **kwargs
        )

        return config_dict, kwargs

    @classmethod
    def _get_config_dict(
        cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        pretrained_model_name_or_path = str(pretrained_model_name_or_path)

        resolved_config_file = try_to_load_from_cache(
            pretrained_model_name_or_path, "config.json"
        )
        logger.info(f"loading configuration file {resolved_config_file}")

        config_dict = cls._dict_from_json_file(resolved_config_file)
        return config_dict, kwargs

    @classmethod
    def _dict_from_json_file(cls, json_file: Union[str, os.PathLike]):
        with open(json_file, encoding="utf-8") as reader:
            text = reader.read()
        import json

        return json.loads(text)

    @property
    def use_return_dict(self) -> bool:
        # If torchscript is set, force `return_dict=False` to avoid jit errors
        return self.return_dict and not self.torchscript

    @property
    def output_attentions(self):
        return self._output_attentions

    @output_attentions.setter
    def output_attentions(self, value: bool):
        # If we set `output_attentions` explicitly before the attn implementation, dispatch eager
        if value and self._attn_implementation is None:
            self._attn_implementation = "eager"
        if value and self._attn_implementation != "eager":
            raise ValueError(
                "The `output_attentions` attribute is not supported when using the `attn_implementation` set to "
                f"{self._attn_implementation}. Please set it to 'eager' instead."
            )
        self._output_attentions = value

    @property
    def _attn_implementation(self):
        return self._attn_implementation_internal

    @_attn_implementation.setter
    def _attn_implementation(self, value: Optional[Union[str, dict]]):
        """We set it recursively on the sub-configs as well"""
        # Set if for current config
        attn_implementation = (
            value
            if not isinstance(value, dict)
            else value.get("", self._attn_implementation)
        )
        self._attn_implementation_internal = attn_implementation

        # Set it recursively on the subconfigs
        for subconfig_key in self.sub_configs:
            subconfig = getattr(self, subconfig_key, None)
            if subconfig is not None:
                sub_implementation = (
                    value
                    if not isinstance(value, dict)
                    else value.get(subconfig_key, subconfig._attn_implementation)
                )
                subconfig._attn_implementation = sub_implementation

    # TODO: check is needed to push the out_feature down the hierarcy
    @property
    def out_features(self):
        return self._out_features

    @out_features.setter
    def out_features(self, out_features: list[str]):
        """
        Set the out_features attribute. This will also update the out_indices attribute to match the new out_features.
        """
        self._out_features, self._out_indices = (
            get_aligned_output_features_output_indices(
                out_features=out_features,
                out_indices=None,
                stage_names=self.stage_names,
            )
        )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, os.PathLike],
        cache_dir: Optional[Union[str, os.PathLike]] = None,
        force_download: bool = False,
        local_files_only: bool = False,
        token: Optional[Union[str, bool]] = None,
        revision: str = "main",
        **kwargs,
    ):
        kwargs["cache_dir"] = cache_dir
        kwargs["force_download"] = force_download
        kwargs["local_files_only"] = local_files_only
        kwargs["revision"] = revision

        config_dict, kwargs = cls.get_config_dict(
            pretrained_model_name_or_path, **kwargs
        )
        return cls.from_dict(config_dict, **kwargs)

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any], **kwargs):
        return_unused_kwargs = kwargs.pop("return_unused_kwargs", False)
        # Those arguments may be passed along for our internal telemetry.
        # We remove them so they don't appear in `return_unused_kwargs`.
        kwargs.pop("_from_auto", None)
        kwargs.pop("_from_pipeline", None)
        # The commit hash might have been updated in the `config_dict`, we don't want the kwargs to erase that update.
        if "_commit_hash" in kwargs and "_commit_hash" in config_dict:
            kwargs["_commit_hash"] = config_dict["_commit_hash"]

        # We remove it from kwargs so that it does not appear in `return_unused_kwargs`.
        config_dict["attn_implementation"] = kwargs.pop("attn_implementation", None)

        config = cls(**config_dict)

        if hasattr(config, "pruned_heads"):
            config.pruned_heads = {
                int(key): value for key, value in config.pruned_heads.items()
            }

        # Update config with kwargs if needed
        if "num_labels" in kwargs and "id2label" in kwargs:
            num_labels = kwargs["num_labels"]
            id2label = kwargs["id2label"] if kwargs["id2label"] is not None else []
            if len(id2label) != num_labels:
                raise ValueError(
                    f"You passed along `num_labels={num_labels}` with an incompatible id to label map: "
                    f"{kwargs['id2label']}. Since those arguments are inconsistent with each other, you should remove "
                    "one of them."
                )
        to_remove = []
        for key, value in kwargs.items():
            if hasattr(config, key):
                current_attr = getattr(config, key)
                # To authorize passing a custom subconfig as kwarg in models that have nested configs.
                if isinstance(current_attr, PretrainedConfig) and isinstance(
                    value, dict
                ):
                    value = current_attr.__class__(**value)
                setattr(config, key, value)
                if key != "torch_dtype":
                    to_remove.append(key)
        for key in to_remove:
            kwargs.pop(key, None)

        logger.info(f"Model config {config}")
        if return_unused_kwargs:
            return config, kwargs
        else:
            return config


class Dinov2Config(LocalPretrainedConfig):
    model_type = "dinov2"

    def __init__(
        self,
        torch_dtype="float32",
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        mlp_ratio=4,
        hidden_act="gelu",
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        initializer_range=0.02,
        layer_norm_eps=1e-6,
        image_size=224,
        patch_size=14,
        num_channels=3,
        qkv_bias=True,
        layerscale_value=1.0,
        drop_path_rate=0.0,
        use_swiglu_ffn=False,
        out_features=None,
        out_indices=None,
        apply_layernorm=True,
        reshape_hidden_states=True,
        use_mask_token=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.mlp_ratio = mlp_ratio
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.qkv_bias = qkv_bias
        self.layerscale_value = layerscale_value
        self.drop_path_rate = drop_path_rate
        self.use_swiglu_ffn = use_swiglu_ffn
        self.stage_names = ["stem"] + [
            f"stage{idx}" for idx in range(1, num_hidden_layers + 1)
        ]
        self._out_features, self._out_indices = (
            get_aligned_output_features_output_indices(
                out_features=out_features,
                out_indices=out_indices,
                stage_names=self.stage_names,
            )
        )
        self.apply_layernorm = apply_layernorm
        self.reshape_hidden_states = reshape_hidden_states
        self.use_mask_token = use_mask_token

        self.torch_dtype = torch_dtype

    def to_dict(self) -> dict[str, Any]:
        output = copy.deepcopy(self.__dict__)
        if hasattr(self.__class__, "model_type"):
            output["model_type"] = self.__class__.model_type

        for key, value in output.items():
            # Deal with nested configs like CLIP
            if isinstance(value, PretrainedConfig):
                value = value.to_dict()
                del value["transformers_version"]

            output[key] = value

        if hasattr(self, "quantization_config"):
            output["quantization_config"] = (
                self.quantization_config.to_dict()
                if not isinstance(self.quantization_config, dict)
                else self.quantization_config
            )

        return output


CONFIG_NAME = "config.json"


def get_configuration_file(configuration_files: list[str]) -> str:
    configuration_files_map = {}
    for file_name in configuration_files:
        if (
            file_name.startswith("config.")
            and file_name.endswith(".json")
            and file_name != "config.json"
        ):
            v = file_name.removeprefix("config.").removesuffix(".json")
            configuration_files_map[v] = file_name

    # Defaults to FULL_CONFIGURATION_FILE and then try to look at some newer versions.
    configuration_file = CONFIG_NAME
    return configuration_file


class LocalPretrainedModel(nn.Module):
    backbone_type: Optional[BackboneType] = None

    def __init__(self, config: LocalPretrainedConfig, *inputs, **kwargs):
        super().__init__()
        if not isinstance(config, LocalPretrainedConfig):
            raise TypeError(
                f"Parameter config in `{self.__class__.__name__}(config)` should be an instance of class "
                "`LocalPretrainedConfig`. To create a model from a pretrained model use "
                f"`model = {self.__class__.__name__}.from_pretrained(PRETRAINED_MODEL_NAME)`"
            )
        self.config = config
        self.config._attn_implementation_internal = (
            self._check_and_adjust_attn_implementation(
                self.config._attn_implementation, is_init_check=True
            )
        )

    def _check_and_adjust_attn_implementation(
        self, attn_implementation: Optional[str], is_init_check: bool = False
    ) -> str:
        # Register kernel if relevant
        attn_implementation = self.get_correct_attn_implementation(
            attn_implementation, is_init_check
        )

        return attn_implementation

    def get_correct_attn_implementation(
        self, requested_attention: Optional[str], is_init_check: bool = False
    ) -> str:
        return requested_attention or "sdpa"

    def _init_backbone(self, config) -> None:
        """
        Method to initialize the backbone. This method is called by the constructor of the base class after the
        pretrained model weights have been loaded.
        """
        self.config = config
        self.backbone_type = BackboneType.TRANSFORMERS
        if self.backbone_type == BackboneType.TRANSFORMERS:
            self._init_transformers_backbone(config)
        else:
            raise ValueError(f"backbone_type {self.backbone_type} not supported.")

    def _init_transformers_backbone(self, config) -> None:
        stage_names = getattr(config, "stage_names")
        out_features = getattr(config, "out_features", None)
        out_indices = getattr(config, "out_indices", None)

        self.stage_names = stage_names
        self._out_features, self._out_indices = (
            get_aligned_output_features_output_indices(
                out_features=out_features,
                out_indices=out_indices,
                stage_names=stage_names,
            )
        )
        # Number of channels for each stage. This is set in the transformer backbone model init
        self.num_features = None

    # TODO: check if should be moved down the hierarcy
    @property
    def out_features(self):
        return self._out_features

    @out_features.setter
    def out_features(self, out_features: list[str]):
        """
        Set the out_features attribute. This will also update the out_indices attribute to match the new out_features.
        """
        self._out_features, self._out_indices = (
            get_aligned_output_features_output_indices(
                out_features=out_features,
                out_indices=None,
                stage_names=self.stage_names,
            )
        )

    def post_init(self):
        self.init_weights()
        self._backward_compatibility_gradient_checkpointing()

        # Make sure the modules correctly exist if the flag is active
        if (
            self._keep_in_fp32_modules is not None
            or self._keep_in_fp32_modules_strict is not None
        ):
            all_parameters = {
                name for name, _ in self.named_parameters() if len(name) > 0
            }
            unique_module_names = set()
            # Get all unique module names in the module graph, without the prefixes
            for param in all_parameters:
                unique_module_names.update(
                    [
                        name
                        for name in param.split(".")
                        if not name.isnumeric() and name not in ["weight", "bias"]
                    ]
                )
            # Check that every module in the keep_in_fp32 list is part of the module graph
            if self._keep_in_fp32_modules is not None:
                for module in self._keep_in_fp32_modules:
                    if module not in unique_module_names:
                        raise ValueError(
                            f"{module} was specified in the `_keep_in_fp32_modules` list, but is not part of the modules in"
                            f" {self.__class__.__name__}"
                        )

            if self._keep_in_fp32_modules_strict is not None:
                for module in self._keep_in_fp32_modules_strict:
                    if module not in unique_module_names:
                        raise ValueError(
                            f"{module} was specified in the `_keep_in_fp32_modules_strict` list, but is not part of the modules in"
                            f" {self.__class__.__name__}"
                        )

        # If current model is a base model, attach `base_model_tp_plan` and `base_model_pp_plan` from config
        self._pp_plan = (
            self.config.base_model_pp_plan.copy()
            if self.config.base_model_pp_plan is not None
            else None
        )

    def get_parameter_or_buffer(self, target: str):
        try:
            return self.get_parameter(target)
        except AttributeError:
            pass
        try:
            return self.get_buffer(target)
        except AttributeError:
            pass
        module, param_name = get_module_from_name(self, target)
        if (
            param_name == "_extra_state"
            and getattr(
                module.__class__, "get_extra_state", torch.nn.Module.get_extra_state
            )
            is not torch.nn.Module.get_extra_state
        ):
            return module.get_extra_state()

        raise AttributeError(
            f"`{target}` is neither a parameter, buffer, nor extra state."
        )

    @classmethod
    def _load_pretrained_model(
        cls,
        model,
        state_dict: Optional[dict],
        checkpoint_files: Optional[list[str]],
        pretrained_model_name_or_path: Optional[str],
        ignore_mismatched_sizes: bool = False,
        device_map: Optional[dict] = None,
        dtype: Optional[torch.dtype] = None,
        key_mapping: Optional[dict[str, str]] = None,
        weights_only: bool = True,
    ):
        original_checkpoint_keys = list(
            load_state_dict(
                checkpoint_files[0], map_location="meta", weights_only=weights_only
            ).keys()
        )
        prefix = model.base_model_prefix
        has_prefix_module = (
            any(s.startswith(prefix) for s in original_checkpoint_keys)
            if len(prefix) > 0
            else False
        )
        expects_prefix_module = hasattr(model, prefix) if len(prefix) > 0 else False
        loading_task_model_from_base_state_dict = (
            not has_prefix_module and expects_prefix_module
        )
        loading_base_model_from_task_state_dict = (
            has_prefix_module and not expects_prefix_module
        )
        key_renaming_mapping = model._get_key_renaming_mapping(
            original_checkpoint_keys,
            key_mapping,
            loading_base_model_from_task_state_dict,
            loading_task_model_from_base_state_dict,
        )

        mismatched_keys, mismatched_shapes = _find_mismatched_keys(
            model,
            state_dict,
            checkpoint_files,
            ignore_mismatched_sizes,
            key_renaming_mapping,
            weights_only,
        )
        key_renaming_mapping = {
            k: v for k, v in key_renaming_mapping.items() if v not in mismatched_keys
        }
        args_list = [
            (shard_file, state_dict, device_map, key_renaming_mapping, model)
            for shard_file in checkpoint_files
        ]
        args_list = tqdm(args_list, desc="Loading checkpoint shards")

        error_msgs = []
        for args in args_list:
            _error_msgs, disk_offload_index, cpu_offload_index = load_shard_file(args)
            error_msgs += _error_msgs

        logger.info(
            f"All the weights of {model.__class__.__name__} were initialized from the model checkpoint at"
            f" {pretrained_model_name_or_path}.\nIf your task is similar to the task the model of the checkpoint"
            f" was trained on, you can already use {model.__class__.__name__} for predictions without further"
            " training."
        )
        return (model,)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
        config: Optional[Union[LocalPretrainedConfig, str, os.PathLike]] = None,
        *model_args,
        **kwargs,
    ):
        state_dict = kwargs.pop("state_dict", None)
        proxies = kwargs.pop("proxies", None)
        from_pipeline = kwargs.pop("_from_pipeline", None)
        from_auto_class = kwargs.pop("_from_auto", False)
        torch_dtype = kwargs.pop("torch_dtype", None)
        device_map = kwargs.pop("device_map", None)
        offload_folder = kwargs.pop("offload_folder", None)
        offload_state_dict = kwargs.pop("offload_state_dict", False)
        subfolder = kwargs.pop("subfolder", "")
        gguf_file = kwargs.pop("gguf_file", None)
        device_mesh = kwargs.pop("device_mesh", None)

        # Load config
        config_path = pretrained_model_name_or_path
        if not isinstance(config, LocalPretrainedConfig):
            config, model_kwargs = cls.config_class.from_pretrained(
                config_path,
                return_unused_kwargs=True,
                proxies=proxies,
                subfolder=subfolder,
                gguf_file=gguf_file,
                _from_auto=from_auto_class,
                _from_pipeline=from_pipeline,
                **kwargs,
            )
        else:
            config = copy.deepcopy(config)
            model_kwargs = kwargs

        transformers_explicit_filename = getattr(config, "transformers_weights", None)

        if transformers_explicit_filename is not None:
            if not transformers_explicit_filename.endswith(
                ".safetensors"
            ) and not transformers_explicit_filename.endswith(
                ".safetensors.index.json"
            ):
                raise ValueError(
                    "The transformers file in the config seems to be incorrect: it is neither a safetensors file "
                    "(*.safetensors) nor a safetensors index file (*.safetensors.index.json): "
                    f"{transformers_explicit_filename}"
                )

        resolved_safetensors_file = try_to_load_from_cache(
            pretrained_model_name_or_path, "model.safetensors"
        )

        logger.info(f"loading safetensors rom file {resolved_safetensors_file}")

        checkpoint_files = [resolved_safetensors_file]
        if checkpoint_files[0].endswith(".safetensors"):
            with safe_open(checkpoint_files[0], framework="pt") as f:
                metadata = f.metadata()
            if metadata.get("format") == "pt":
                pass
            else:
                raise ValueError(
                    f"Incompatible safetensors file. File metadata is not ['pt'] but {metadata.get('format')}"
                )

        torch_dtype = torch.float32
        # Find the correct dtype based on current state
        config.name_or_path = pretrained_model_name_or_path
        model = cls(config, *model_args, **model_kwargs)
        config = model.config
        dtype_orig = torch.float16
        if True:
            # restore default dtype
            if dtype_orig is not None:
                torch.set_default_dtype(dtype_orig)

            (model,) = cls._load_pretrained_model(
                model,
                state_dict,
                checkpoint_files,
                pretrained_model_name_or_path,
                dtype=torch_dtype,
            )
        # make sure token embedding weights are still tied if needed
        # model.tie_weights()

        # Set model in evaluation mode to deactivate DropOut modules by default
        model.eval()
        return model


class DepthAnythingConfig(LocalPretrainedConfig):
    model_type = "depth_anything"

    def __init__(
        self,
        backbone_config=None,
        backbone=None,
        use_pretrained_backbone=False,
        use_timm_backbone=False,
        backbone_kwargs=None,
        patch_size=14,
        initializer_range=0.02,
        reassemble_hidden_size=384,
        reassemble_factors=[4, 2, 1, 0.5],
        neck_hidden_sizes=[48, 96, 192, 384],
        fusion_hidden_size=64,
        head_in_index=-1,
        head_hidden_size=32,
        depth_estimation_type="relative",
        max_depth=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if backbone_config is None and backbone is None:
            logger.info(
                "`backbone_config` is `None`. Initializing the config with the default `Dinov2` backbone."
            )
            backbone_config = Dinov2Config(
                image_size=518,
                hidden_size=384,
                num_attention_heads=6,
                out_indices=[9, 10, 11, 12],
                apply_layernorm=True,
                reshape_hidden_states=False,
            )
        elif isinstance(backbone_config, dict):
            backbone_config = Dinov2Config(**backbone_config)

        self.backbone_config = backbone_config
        self.backbone = backbone
        self.use_pretrained_backbone = use_pretrained_backbone
        self.use_timm_backbone = use_timm_backbone
        self.backbone_kwargs = backbone_kwargs
        self.reassemble_hidden_size = reassemble_hidden_size
        self.patch_size = patch_size
        self.initializer_range = initializer_range
        self.reassemble_factors = reassemble_factors
        self.neck_hidden_sizes = neck_hidden_sizes
        self.fusion_hidden_size = fusion_hidden_size
        self.head_in_index = head_in_index
        self.head_hidden_size = head_hidden_size
        if depth_estimation_type not in ["relative", "metric"]:
            raise ValueError(
                "depth_estimation_type must be one of ['relative', 'metric']"
            )
        self.depth_estimation_type = depth_estimation_type
        self.max_depth = max_depth if max_depth else 1

    @property
    def sub_configs(self):
        return (
            {"backbone_config": type(self.backbone_config)}
            if getattr(self, "backbone_config", None) is not None
            else {}
        )

    def to_dict(self):
        output = copy.deepcopy(self.__dict__)

        if output["backbone_config"] is not None:
            output["backbone_config"] = self.backbone_config.to_dict()

        output["model_type"] = self.__class__.model_type
        return output


class DepthAnythingPreTrainedModel(LocalPretrainedModel):
    base_model_prefix = "depth_anything"
    main_input_name = "pixel_values"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    @staticmethod
    def _fix_state_dict_key_on_load(key: str) -> tuple[str, bool]:
        """Replace legacy parameter names with their modern equivalents. E.g. beta -> bias, gamma -> weight."""
        # Rename LayerNorm beta & gamma params for some early models ported from Tensorflow (e.g. Bert)
        # This rename is logged.
        if key.endswith("LayerNorm.beta"):
            return key.replace("LayerNorm.beta", "LayerNorm.bias"), True
        if key.endswith("LayerNorm.gamma"):
            return key.replace("LayerNorm.gamma", "LayerNorm.weight"), True

        # Rename weight norm parametrizations to match changes across torch versions.
        # Impacts a number of speech/wav2vec models. e.g. Hubert, Wav2Vec2, and others.
        # This rename is not logged.
        if hasattr(nn.utils.parametrizations, "weight_norm"):
            if key.endswith("weight_g"):
                return key.replace(
                    "weight_g", "parametrizations.weight.original0"
                ), True
            if key.endswith("weight_v"):
                return key.replace(
                    "weight_v", "parametrizations.weight.original1"
                ), True
        else:
            if key.endswith("parametrizations.weight.original0"):
                return key.replace(
                    "parametrizations.weight.original0", "weight_g"
                ), True
            if key.endswith("parametrizations.weight.original1"):
                return key.replace(
                    "parametrizations.weight.original1", "weight_v"
                ), True

        return key, False

    def _get_key_renaming_mapping(
        self,
        checkpoint_keys: list[str],
        key_mapping: Optional[dict[str, str]] = None,
        loading_base_model_from_task_state_dict: bool = False,
        loading_task_model_from_base_state_dict: bool = False,
    ):
        prefix = self.base_model_prefix
        _prefix = f"{prefix}."

        renamed_keys = {}
        key_renaming_mapping = {}
        for key in checkpoint_keys:
            # Class specific rename
            new_key, has_changed = self._fix_state_dict_key_on_load(key)

            # Optionally map the key according to `key_mapping`
            if key_mapping is not None:
                for pattern, replacement in key_mapping.items():
                    new_key, n_replace = re.subn(pattern, replacement, new_key)
                    # Early exit of the loop
                    if n_replace > 0:
                        has_changed = True
                        break

            # In this case, we need to add the prefix to the keys, to match them to the expected keys
            if loading_task_model_from_base_state_dict:
                new_key = ".".join([prefix, new_key])
            # In this case we need to remove the prefix from the key to match them to the expected keys, and use
            # only the keys starting with the prefix
            elif loading_base_model_from_task_state_dict:
                if not new_key.startswith(_prefix):
                    continue
                new_key = new_key[len(_prefix) :]

            key_renaming_mapping[key] = new_key

            # track gamma/beta rename for logging
            if has_changed:
                if key.endswith("LayerNorm.gamma"):
                    renamed_keys["LayerNorm.gamma"] = (key, new_key)
                elif key.endswith("LayerNorm.beta"):
                    renamed_keys["LayerNorm.beta"] = (key, new_key)

        if renamed_keys:
            warning_msg = f"A pretrained model of type `{self.__class__.__name__}` "
            warning_msg += "contains parameters that have been renamed internally (a few are listed below but more are present in the model):\n"
            for old_key, new_key in renamed_keys.values():
                warning_msg += f"* `{old_key}` -> `{new_key}`\n"
            warning_msg += "If you are using a model from the Hub, consider submitting a PR to adjust these weights and help future users."
            logger.info_once(warning_msg)

        return key_renaming_mapping


class ModelOutput(OrderedDict):
    def __init_subclass__(cls) -> None:
        if is_torch_available():
            if version.parse(get_torch_version()) >= version.parse("2.2"):
                from torch.utils._pytree import register_pytree_node

                register_pytree_node(
                    cls,
                    _model_output_flatten,
                    partial(_model_output_unflatten, output_type=cls),
                    serialized_type_name=f"{cls.__module__}.{cls.__name__}",
                )
            else:
                from torch.utils._pytree import _register_pytree_node

                _register_pytree_node(
                    cls,
                    _model_output_flatten,
                    partial(_model_output_unflatten, output_type=cls),
                )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Subclasses of ModelOutput must use the @dataclass decorator
        # This check is done in __init__ because the @dataclass decorator operates after __init_subclass__
        # issubclass() would return True for issubclass(ModelOutput, ModelOutput) when False is needed
        # Just need to check that the current class is not ModelOutput
        is_modeloutput_subclass = self.__class__ != ModelOutput

        if is_modeloutput_subclass and not is_dataclass(self):
            raise TypeError(
                f"{self.__module__}.{self.__class__.__name__} is not a dataclass."
                " This is a subclass of ModelOutput and so must use the @dataclass decorator."
            )

    def __post_init__(self):
        """Check the ModelOutput dataclass.

        Only occurs if @dataclass decorator has been used.
        """
        class_fields = fields(self)

        # Safety and consistency checks
        if not len(class_fields):
            raise ValueError(f"{self.__class__.__name__} has no fields.")
        if not all(field.default is None for field in class_fields[1:]):
            raise ValueError(
                f"{self.__class__.__name__} should not have more than one required field."
            )

        first_field = getattr(self, class_fields[0].name)
        other_fields_are_none = all(
            getattr(self, field.name) is None for field in class_fields[1:]
        )

        if other_fields_are_none and not is_tensor(first_field):
            if isinstance(first_field, dict):
                iterator = first_field.items()
                first_field_iterator = True
            else:
                try:
                    iterator = iter(first_field)
                    first_field_iterator = True
                except TypeError:
                    first_field_iterator = False

            # if we provided an iterator as first field and the iterator is a (key, value) iterator
            # set the associated fields
            if first_field_iterator:
                for idx, element in enumerate(iterator):
                    if (
                        not isinstance(element, (list, tuple))
                        or len(element) != 2
                        or not isinstance(element[0], str)
                    ):
                        if idx == 0:
                            # If we do not have an iterator of key/values, set it as attribute
                            self[class_fields[0].name] = first_field
                        else:
                            # If we have a mixed iterator, raise an error
                            raise ValueError(
                                f"Cannot set key/value for {element}. It needs to be a tuple (key, value)."
                            )
                        break
                    setattr(self, element[0], element[1])
                    if element[1] is not None:
                        self[element[0]] = element[1]
            elif first_field is not None:
                self[class_fields[0].name] = first_field
        else:
            for field in class_fields:
                v = getattr(self, field.name)
                if v is not None:
                    self[field.name] = v

    def __delitem__(self, *args, **kwargs):
        raise Exception(
            f"You cannot use ``__delitem__`` on a {self.__class__.__name__} instance."
        )

    def setdefault(self, *args, **kwargs):
        raise Exception(
            f"You cannot use ``setdefault`` on a {self.__class__.__name__} instance."
        )

    def pop(self, *args, **kwargs):
        raise Exception(
            f"You cannot use ``pop`` on a {self.__class__.__name__} instance."
        )

    def update(self, *args, **kwargs):
        raise Exception(
            f"You cannot use ``update`` on a {self.__class__.__name__} instance."
        )

    def __getitem__(self, k):
        if isinstance(k, str):
            inner_dict = dict(self.items())
            return inner_dict[k]
        else:
            return self.to_tuple()[k]

    def __setattr__(self, name, value):
        if name in self.keys() and value is not None:
            # Don't call self.__setitem__ to avoid recursion errors
            super().__setitem__(name, value)
        super().__setattr__(name, value)

    def __setitem__(self, key, value):
        # Will raise a KeyException if needed
        super().__setitem__(key, value)
        # Don't call self.__setattr__ to avoid recursion errors
        super().__setattr__(key, value)

    def __reduce__(self):
        if not is_dataclass(self):
            return super().__reduce__()
        callable, _args, *remaining = super().__reduce__()
        args = tuple(getattr(self, field.name) for field in fields(self))
        return callable, args, *remaining

    def to_tuple(self) -> tuple[Any]:
        """
        Convert self to a tuple containing all the attributes/keys that are not `None`.
        """
        return tuple(self[k] for k in self.keys())


if is_torch_available():
    import torch.utils._pytree as _torch_pytree

    def _model_output_flatten(
        output: ModelOutput,
    ) -> tuple[list[Any], "_torch_pytree.Context"]:
        return list(output.values()), list(output.keys())

    def _model_output_unflatten(
        values: Iterable[Any],
        context: "_torch_pytree.Context",
        output_type=None,
    ) -> ModelOutput:
        return output_type(**dict(zip(context, values)))

    if version.parse(get_torch_version()) >= version.parse("2.2"):
        _torch_pytree.register_pytree_node(
            ModelOutput,
            _model_output_flatten,
            partial(_model_output_unflatten, output_type=ModelOutput),
            serialized_type_name=f"{ModelOutput.__module__}.{ModelOutput.__name__}",
        )
    else:
        _torch_pytree._register_pytree_node(
            ModelOutput,
            _model_output_flatten,
            partial(_model_output_unflatten, output_type=ModelOutput),
        )


@dataclass
class DepthEstimatorOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    predicted_depth: Optional[torch.FloatTensor] = None
    hidden_states: Optional[tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[tuple[torch.FloatTensor, ...]] = None


class DepthAnythingReassembleLayer(nn.Module):
    def __init__(self, config, channels, factor):
        super().__init__()
        self.projection = nn.Conv2d(
            in_channels=config.reassemble_hidden_size,
            out_channels=channels,
            kernel_size=1,
        )

        # up/down sampling depending on factor
        if factor > 1:
            self.resize = nn.ConvTranspose2d(
                channels, channels, kernel_size=factor, stride=factor, padding=0
            )
        elif factor == 1:
            self.resize = nn.Identity()
        elif factor < 1:
            # so should downsample
            self.resize = nn.Conv2d(
                channels, channels, kernel_size=3, stride=int(1 / factor), padding=1
            )

    # Copied from transformers.models.dpt.modeling_dpt.DPTReassembleLayer.forward
    def forward(self, hidden_state):
        hidden_state = self.projection(hidden_state)
        hidden_state = self.resize(hidden_state)

        return hidden_state


class DepthAnythingReassembleStage(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.layers = nn.ModuleList()
        for channels, factor in zip(
            config.neck_hidden_sizes, config.reassemble_factors
        ):
            self.layers.append(
                DepthAnythingReassembleLayer(config, channels=channels, factor=factor)
            )

    def forward(
        self, hidden_states: list[torch.Tensor], patch_height=None, patch_width=None
    ) -> list[torch.Tensor]:
        out = []

        for i, hidden_state in enumerate(hidden_states):
            # reshape to (batch_size, num_channels, height, width)
            hidden_state = hidden_state[:, 1:]
            batch_size, _, num_channels = hidden_state.shape
            hidden_state = hidden_state.reshape(
                batch_size, patch_height, patch_width, num_channels
            )
            hidden_state = hidden_state.permute(0, 3, 1, 2).contiguous()
            hidden_state = self.layers[i](hidden_state)
            out.append(hidden_state)

        return out


class DepthAnythingPreActResidualLayer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.activation1 = nn.ReLU()
        self.convolution1 = nn.Conv2d(
            config.fusion_hidden_size,
            config.fusion_hidden_size,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )

        self.activation2 = nn.ReLU()
        self.convolution2 = nn.Conv2d(
            config.fusion_hidden_size,
            config.fusion_hidden_size,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        residual = hidden_state
        hidden_state = self.activation1(hidden_state)
        hidden_state = self.convolution1(hidden_state)
        hidden_state = self.activation2(hidden_state)
        hidden_state = self.convolution2(hidden_state)

        return hidden_state + residual


class DepthAnythingFeatureFusionLayer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.projection = nn.Conv2d(
            config.fusion_hidden_size,
            config.fusion_hidden_size,
            kernel_size=1,
            bias=True,
        )

        self.residual_layer1 = DepthAnythingPreActResidualLayer(config)
        self.residual_layer2 = DepthAnythingPreActResidualLayer(config)

    def forward(self, hidden_state, residual=None, size=None):
        if residual is not None:
            if hidden_state.shape != residual.shape:
                residual = nn.functional.interpolate(
                    residual,
                    size=(hidden_state.shape[2], hidden_state.shape[3]),
                    mode="bilinear",
                    align_corners=False,
                )
            hidden_state = hidden_state + self.residual_layer1(residual)

        hidden_state = self.residual_layer2(hidden_state)

        modifier = {"scale_factor": 2} if size is None else {"size": size}

        hidden_state = nn.functional.interpolate(
            hidden_state,
            **modifier,
            mode="bilinear",
            align_corners=True,
        )
        hidden_state = self.projection(hidden_state)

        return hidden_state


class DepthAnythingFeatureFusionStage(nn.Module):
    # Copied from transformers.models.dpt.modeling_dpt.DPTFeatureFusionStage.__init__ with DPT->DepthAnything
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(len(config.neck_hidden_sizes)):
            self.layers.append(DepthAnythingFeatureFusionLayer(config))

    def forward(self, hidden_states, size=None):
        # reversing the hidden_states, we start from the last
        hidden_states = hidden_states[::-1]

        fused_hidden_states = []
        fused_hidden_state = None

        for idx, (hidden_state, layer) in enumerate(zip(hidden_states, self.layers)):
            size = (
                hidden_states[idx + 1].shape[2:]
                if idx != (len(hidden_states) - 1)
                else None
            )

            if fused_hidden_state is None:
                # first layer only uses the last hidden_state
                fused_hidden_state = layer(hidden_state, size=size)
            else:
                fused_hidden_state = layer(fused_hidden_state, hidden_state, size=size)

            fused_hidden_states.append(fused_hidden_state)

        return fused_hidden_states


class DepthAnythingNeck(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.reassemble_stage = DepthAnythingReassembleStage(config)

        self.convs = nn.ModuleList()
        for channel in config.neck_hidden_sizes:
            self.convs.append(
                nn.Conv2d(
                    channel,
                    config.fusion_hidden_size,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                )
            )

        # fusion
        self.fusion_stage = DepthAnythingFeatureFusionStage(config)

    def forward(
        self, hidden_states: list[torch.Tensor], patch_height=None, patch_width=None
    ) -> list[torch.Tensor]:
        if not isinstance(hidden_states, (tuple, list)):
            raise TypeError("hidden_states should be a tuple or list of tensors")

        if len(hidden_states) != len(self.config.neck_hidden_sizes):
            raise ValueError(
                "The number of hidden states should be equal to the number of neck hidden sizes."
            )

        # postprocess hidden states
        hidden_states = self.reassemble_stage(hidden_states, patch_height, patch_width)

        features = [self.convs[i](feature) for i, feature in enumerate(hidden_states)]

        # fusion blocks
        output = self.fusion_stage(features)

        return output


class DepthAnythingDepthEstimationHead(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.head_in_index = config.head_in_index
        self.patch_size = config.patch_size

        features = config.fusion_hidden_size
        self.conv1 = nn.Conv2d(
            features, features // 2, kernel_size=3, stride=1, padding=1
        )
        self.conv2 = nn.Conv2d(
            features // 2, config.head_hidden_size, kernel_size=3, stride=1, padding=1
        )
        self.activation1 = nn.ReLU()
        self.conv3 = nn.Conv2d(
            config.head_hidden_size, 1, kernel_size=1, stride=1, padding=0
        )
        if config.depth_estimation_type == "relative":
            self.activation2 = nn.ReLU()
        elif config.depth_estimation_type == "metric":
            self.activation2 = nn.Sigmoid()
        else:
            raise ValueError(
                f"Unknown depth estimation type: {config.depth_estimation_type}"
            )
        self.max_depth = config.max_depth

    def forward(
        self, hidden_states: list[torch.Tensor], patch_height, patch_width
    ) -> torch.Tensor:
        hidden_states = hidden_states[self.head_in_index]

        predicted_depth = self.conv1(hidden_states)
        predicted_depth = nn.functional.interpolate(
            predicted_depth,
            (int(patch_height * self.patch_size), int(patch_width * self.patch_size)),
            mode="bilinear",
            align_corners=True,
        )
        predicted_depth = self.conv2(predicted_depth)
        predicted_depth = self.activation1(predicted_depth)
        predicted_depth = self.conv3(predicted_depth)
        predicted_depth = self.activation2(predicted_depth) * self.max_depth
        predicted_depth = predicted_depth.squeeze(
            dim=1
        )  # shape (batch_size, height, width)

        return predicted_depth


class Dinov2Embeddings(nn.Module):
    """
    Construct the CLS token, mask token, position and patch embeddings.
    """

    def __init__(self, config: Dinov2Config) -> None:
        super().__init__()

        self.cls_token = nn.Parameter(torch.randn(1, 1, config.hidden_size))
        if config.use_mask_token:
            self.mask_token = nn.Parameter(torch.zeros(1, config.hidden_size))
        self.patch_embeddings = Dinov2PatchEmbeddings(config)
        num_patches = self.patch_embeddings.num_patches
        self.position_embeddings = nn.Parameter(
            torch.randn(1, num_patches + 1, config.hidden_size)
        )
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.patch_size = config.patch_size
        self.use_mask_token = config.use_mask_token
        self.config = config

    def interpolate_pos_encoding(
        self, embeddings: torch.Tensor, height: int, width: int
    ) -> torch.Tensor:
        num_patches = embeddings.shape[1] - 1
        num_positions = self.position_embeddings.shape[1] - 1

        # always interpolate when tracing to ensure the exported model works for dynamic input shapes
        if (
            not torch.jit.is_tracing()
            and num_patches == num_positions
            and height == width
        ):
            return self.position_embeddings

        class_pos_embed = self.position_embeddings[:, :1]
        patch_pos_embed = self.position_embeddings[:, 1:]

        dim = embeddings.shape[-1]

        new_height = height // self.patch_size
        new_width = width // self.patch_size

        sqrt_num_positions = torch_int(num_positions**0.5)
        patch_pos_embed = patch_pos_embed.reshape(
            1, sqrt_num_positions, sqrt_num_positions, dim
        )
        patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)
        target_dtype = patch_pos_embed.dtype
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.to(torch.float32),
            size=(new_height, new_width),
            mode="bicubic",
            align_corners=False,
        ).to(dtype=target_dtype)

        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)

        return torch.cat((class_pos_embed, patch_pos_embed), dim=1)

    def forward(
        self, pixel_values: torch.Tensor, bool_masked_pos: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size, _, height, width = pixel_values.shape
        target_dtype = self.patch_embeddings.projection.weight.dtype
        embeddings = self.patch_embeddings(pixel_values.to(dtype=target_dtype))

        if bool_masked_pos is not None and self.use_mask_token:
            embeddings = torch.where(
                bool_masked_pos.unsqueeze(-1),
                self.mask_token.to(embeddings.dtype).unsqueeze(0),
                embeddings,
            )

        # add the [CLS] token to the embedded patch tokens
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)

        # add positional encoding to each token
        embeddings = embeddings + self.interpolate_pos_encoding(
            embeddings, height, width
        )

        embeddings = self.dropout(embeddings)

        return embeddings


class Dinov2PatchEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        image_size, patch_size = config.image_size, config.patch_size
        num_channels, hidden_size = config.num_channels, config.hidden_size

        image_size = (
            image_size
            if isinstance(image_size, collections.abc.Iterable)
            else (image_size, image_size)
        )
        patch_size = (
            patch_size
            if isinstance(patch_size, collections.abc.Iterable)
            else (patch_size, patch_size)
        )
        num_patches = (image_size[1] // patch_size[1]) * (
            image_size[0] // patch_size[0]
        )
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_patches = num_patches

        self.projection = nn.Conv2d(
            num_channels, hidden_size, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        num_channels = pixel_values.shape[1]
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
                f" Expected {self.num_channels} but got {num_channels}."
            )
        embeddings = self.projection(pixel_values).flatten(2).transpose(1, 2)
        return embeddings


class GradientCheckpointingLayer(nn.Module):
    gradient_checkpointing = False

    def __call__(self, *args, **kwargs):
        if self.gradient_checkpointing and self.training:
            do_warn = False
            layer_name = self.__class__.__name__
            message = f"Caching is incompatible with gradient checkpointing in {layer_name}. Setting"

            if "use_cache" in kwargs and kwargs["use_cache"]:
                kwargs["use_cache"] = False
                message += " `use_cache=False`,"
                do_warn = True

            # different names for the same thing in different layers
            # TODO cyril: this one without `S` can be removed after deprection cycle
            if "past_key_value" in kwargs and kwargs["past_key_value"] is not None:
                kwargs["past_key_value"] = None
                message += " `past_key_value=None`,"
                do_warn = True

            if "past_key_values" in kwargs and kwargs["past_key_values"] is not None:
                kwargs["past_key_values"] = None
                message += " `past_key_values=None`,"
                do_warn = True

            if "layer_past" in kwargs and kwargs["layer_past"] is not None:
                kwargs["layer_past"] = None
                message += " `layer_past=None`,"
                do_warn = True

            # warn if anything was changed
            if do_warn:
                message = message.rstrip(",") + "."
                logger.warning_once(message)

            return self._gradient_checkpointing_func(
                partial(super().__call__, **kwargs), *args
            )
        return super().__call__(*args, **kwargs)


# Copied from transformers.models.vit.modeling_vit.eager_attention_forward
def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    # Take the dot product between "query" and "key" to get the raw attention scores.
    attn_weights = torch.matmul(query, key.transpose(-1, -2)) * scaling

    # Normalize the attention scores to probabilities.
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
        query.dtype
    )

    # This is actually dropping out entire tokens to attend to, which might
    # seem a bit unusual, but is taken from the original Transformer paper.
    attn_weights = nn.functional.dropout(
        attn_weights, p=dropout, training=module.training
    )

    # Mask heads if we want to
    if attention_mask is not None:
        attn_weights = attn_weights * attention_mask

    attn_output = torch.matmul(attn_weights, value)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class Dinov2SelfAttention(nn.Module):
    def __init__(self, config: Dinov2Config) -> None:
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(
            config, "embedding_size"
        ):
            raise ValueError(
                f"The hidden size {config.hidden_size} is not a multiple of the number of attention "
                f"heads {config.num_attention_heads}."
            )

        self.config = config
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.dropout_prob = config.attention_probs_dropout_prob
        self.scaling = self.attention_head_size**-0.5
        self.is_causal = False

        self.query = nn.Linear(
            config.hidden_size, self.all_head_size, bias=config.qkv_bias
        )
        self.key = nn.Linear(
            config.hidden_size, self.all_head_size, bias=config.qkv_bias
        )
        self.value = nn.Linear(
            config.hidden_size, self.all_head_size, bias=config.qkv_bias
        )

    def forward(
        self,
        hidden_states,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor]]:
        batch_size, seq_length, _ = hidden_states.shape
        key_layer = (
            self.key(hidden_states)
            .view(batch_size, -1, self.num_attention_heads, self.attention_head_size)
            .transpose(1, 2)
        )
        value_layer = (
            self.value(hidden_states)
            .view(batch_size, -1, self.num_attention_heads, self.attention_head_size)
            .transpose(1, 2)
        )
        query_layer = (
            self.query(hidden_states)
            .view(batch_size, -1, self.num_attention_heads, self.attention_head_size)
            .transpose(1, 2)
        )

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa" and output_attentions:
                logger.warning_once(
                    "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                    'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
                )
            else:
                attention_interface = ALL_ATTENTION_FUNCTIONS[
                    self.config._attn_implementation
                ]

        context_layer, attention_probs = attention_interface(
            self,
            query_layer,
            key_layer,
            value_layer,
            head_mask,
            is_causal=self.is_causal,
            scaling=self.scaling,
            dropout=0.0 if not self.training else self.dropout_prob,
        )

        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.reshape(new_context_layer_shape)

        outputs = (
            (context_layer, attention_probs) if output_attentions else (context_layer,)
        )

        return outputs


class Dinov2SelfOutput(nn.Module):
    def __init__(self, config: Dinov2Config) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self, hidden_states: torch.Tensor, input_tensor: torch.Tensor
    ) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return hidden_states


from transformers.activations import ACT2FN


class Dinov2MLP(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        in_features = out_features = config.hidden_size
        hidden_features = int(config.hidden_size * config.mlp_ratio)
        self.fc1 = nn.Linear(in_features, hidden_features, bias=True)
        if isinstance(config.hidden_act, str):
            self.activation = ACT2FN[config.hidden_act]
        else:
            self.activation = config.hidden_act
        self.fc2 = nn.Linear(hidden_features, out_features, bias=True)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        hidden_state = self.fc1(hidden_state)
        hidden_state = self.activation(hidden_state)
        hidden_state = self.fc2(hidden_state)
        return hidden_state


class Dinov2LayerScale(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.lambda1 = nn.Parameter(
            config.layerscale_value * torch.ones(config.hidden_size)
        )

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        return hidden_state * self.lambda1


class Dinov2Attention(nn.Module):
    def __init__(self, config: Dinov2Config) -> None:
        super().__init__()
        self.attention = Dinov2SelfAttention(config)
        self.output = Dinov2SelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads: set[int]) -> None:
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads,
            self.attention.num_attention_heads,
            self.attention.attention_head_size,
            self.pruned_heads,
        )

        # Prune linear layers
        self.attention.query = prune_linear_layer(self.attention.query, index)
        self.attention.key = prune_linear_layer(self.attention.key, index)
        self.attention.value = prune_linear_layer(self.attention.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.attention.num_attention_heads = self.attention.num_attention_heads - len(
            heads
        )
        self.attention.all_head_size = (
            self.attention.attention_head_size * self.attention.num_attention_heads
        )
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor]]:
        self_outputs = self.attention(hidden_states, head_mask, output_attentions)

        attention_output = self.output(self_outputs[0], hidden_states)

        outputs = (attention_output,) + self_outputs[
            1:
        ]  # add attentions if we output them
        return outputs


class Dinov2Layer(GradientCheckpointingLayer):
    """This corresponds to the Block class in the original implementation."""

    def __init__(self, config: Dinov2Config) -> None:
        super().__init__()

        self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attention = Dinov2Attention(config)
        self.layer_scale1 = Dinov2LayerScale(config)
        self.drop_path = (
            Dinov2DropPath(config.drop_path_rate)
            if config.drop_path_rate > 0.0
            else nn.Identity()
        )

        self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.mlp = Dinov2MLP(config)
        self.layer_scale2 = Dinov2LayerScale(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor]]:
        self_attention_outputs = self.attention(
            self.norm1(
                hidden_states
            ),  # in Dinov2, layernorm is applied before self-attention
            head_mask,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]

        attention_output = self.layer_scale1(attention_output)
        outputs = self_attention_outputs[
            1:
        ]  # add self attentions if we output attention weights

        # first residual connection
        hidden_states = self.drop_path(attention_output) + hidden_states

        # in Dinov2, layernorm is also applied after self-attention
        layer_output = self.norm2(hidden_states)
        layer_output = self.mlp(layer_output)
        layer_output = self.layer_scale2(layer_output)

        # second residual connection
        layer_output = self.drop_path(layer_output) + hidden_states

        outputs = (layer_output,) + outputs

        return outputs


@dataclass
class BackboneOutput(ModelOutput):
    feature_maps: Optional[tuple[torch.FloatTensor]] = None
    hidden_states: Optional[tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[tuple[torch.FloatTensor, ...]] = None


class ModelOutput(OrderedDict):
    def __init_subclass__(cls) -> None:
        if is_torch_available():
            if version.parse(get_torch_version()) >= version.parse("2.2"):
                from torch.utils._pytree import register_pytree_node

                register_pytree_node(
                    cls,
                    _model_output_flatten,
                    partial(_model_output_unflatten, output_type=cls),
                    serialized_type_name=f"{cls.__module__}.{cls.__name__}",
                )
            else:
                from torch.utils._pytree import _register_pytree_node

                _register_pytree_node(
                    cls,
                    _model_output_flatten,
                    partial(_model_output_unflatten, output_type=cls),
                )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Subclasses of ModelOutput must use the @dataclass decorator
        # This check is done in __init__ because the @dataclass decorator operates after __init_subclass__
        # issubclass() would return True for issubclass(ModelOutput, ModelOutput) when False is needed
        # Just need to check that the current class is not ModelOutput
        is_modeloutput_subclass = self.__class__ != ModelOutput

        if is_modeloutput_subclass and not is_dataclass(self):
            raise TypeError(
                f"{self.__module__}.{self.__class__.__name__} is not a dataclass."
                " This is a subclass of ModelOutput and so must use the @dataclass decorator."
            )

    def __post_init__(self):
        """Check the ModelOutput dataclass.

        Only occurs if @dataclass decorator has been used.
        """
        class_fields = fields(self)

        # Safety and consistency checks
        if not len(class_fields):
            raise ValueError(f"{self.__class__.__name__} has no fields.")
        if not all(field.default is None for field in class_fields[1:]):
            raise ValueError(
                f"{self.__class__.__name__} should not have more than one required field."
            )

        first_field = getattr(self, class_fields[0].name)
        other_fields_are_none = all(
            getattr(self, field.name) is None for field in class_fields[1:]
        )

        if other_fields_are_none and not is_tensor(first_field):
            if isinstance(first_field, dict):
                iterator = first_field.items()
                first_field_iterator = True
            else:
                try:
                    iterator = iter(first_field)
                    first_field_iterator = True
                except TypeError:
                    first_field_iterator = False

            # if we provided an iterator as first field and the iterator is a (key, value) iterator
            # set the associated fields
            if first_field_iterator:
                for idx, element in enumerate(iterator):
                    if (
                        not isinstance(element, (list, tuple))
                        or len(element) != 2
                        or not isinstance(element[0], str)
                    ):
                        if idx == 0:
                            # If we do not have an iterator of key/values, set it as attribute
                            self[class_fields[0].name] = first_field
                        else:
                            # If we have a mixed iterator, raise an error
                            raise ValueError(
                                f"Cannot set key/value for {element}. It needs to be a tuple (key, value)."
                            )
                        break
                    setattr(self, element[0], element[1])
                    if element[1] is not None:
                        self[element[0]] = element[1]
            elif first_field is not None:
                self[class_fields[0].name] = first_field
        else:
            for field in class_fields:
                v = getattr(self, field.name)
                if v is not None:
                    self[field.name] = v

    def __delitem__(self, *args, **kwargs):
        raise Exception(
            f"You cannot use ``__delitem__`` on a {self.__class__.__name__} instance."
        )

    def setdefault(self, *args, **kwargs):
        raise Exception(
            f"You cannot use ``setdefault`` on a {self.__class__.__name__} instance."
        )

    def pop(self, *args, **kwargs):
        raise Exception(
            f"You cannot use ``pop`` on a {self.__class__.__name__} instance."
        )

    def update(self, *args, **kwargs):
        raise Exception(
            f"You cannot use ``update`` on a {self.__class__.__name__} instance."
        )

    def __getitem__(self, k):
        if isinstance(k, str):
            inner_dict = dict(self.items())
            return inner_dict[k]
        else:
            return self.to_tuple()[k]

    def __setattr__(self, name, value):
        if name in self.keys() and value is not None:
            # Don't call self.__setitem__ to avoid recursion errors
            super().__setitem__(name, value)
        super().__setattr__(name, value)

    def __setitem__(self, key, value):
        # Will raise a KeyException if needed
        super().__setitem__(key, value)
        # Don't call self.__setattr__ to avoid recursion errors
        super().__setattr__(key, value)

    def __reduce__(self):
        if not is_dataclass(self):
            return super().__reduce__()
        callable, _args, *remaining = super().__reduce__()
        args = tuple(getattr(self, field.name) for field in fields(self))
        return callable, args, *remaining

    def to_tuple(self) -> tuple[Any]:
        return tuple(self[k] for k in self.keys())


@dataclass
class BaseModelOutput(ModelOutput):
    last_hidden_state: Optional[torch.FloatTensor] = None
    hidden_states: Optional[tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[tuple[torch.FloatTensor, ...]] = None


class Dinov2Encoder(nn.Module):
    def __init__(self, config: Dinov2Config) -> None:
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList(
            [Dinov2Layer(config) for _ in range(config.num_hidden_layers)]
        )
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Union[tuple, BaseModelOutput]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None

            layer_outputs = layer_module(
                hidden_states, layer_head_mask, output_attentions
            )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, all_hidden_states, all_self_attentions]
                if v is not None
            )
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class Dinov2PreTrainedModel(LocalPretrainedModel):
    config: Dinov2Config
    base_model_prefix = "dinov2"
    main_input_name = "pixel_values"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Dinov2Layer"]
    _supports_sdpa = True
    _supports_flash_attn = True
    _supports_flex_attn = True
    _supports_attention_backend = True

    def _init_weights(self, module: Union[nn.Linear, nn.Conv2d, nn.LayerNorm]) -> None:
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # Upcast the input in `fp32` and cast it back to desired `dtype` to avoid
            # `trunc_normal_cpu` not implemented in `half` issues
            module.weight.data = nn.init.trunc_normal_(
                module.weight.data.to(torch.float32),
                mean=0.0,
                std=self.config.initializer_range,
            ).to(module.weight.dtype)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, Dinov2Embeddings):
            module.position_embeddings.data = nn.init.trunc_normal_(
                module.position_embeddings.data.to(torch.float32),
                mean=0.0,
                std=self.config.initializer_range,
            ).to(module.position_embeddings.dtype)

            module.cls_token.data = nn.init.trunc_normal_(
                module.cls_token.data.to(torch.float32),
                mean=0.0,
                std=self.config.initializer_range,
            ).to(module.cls_token.dtype)

            if self.config.use_mask_token:
                module.mask_token.data.zero_()
        elif isinstance(module, Dinov2LayerScale):
            module.lambda1.data.fill_(self.config.layerscale_value)


class Dinov2Backbone(Dinov2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        super()._init_backbone(config)
        self.num_features = [
            config.hidden_size for _ in range(config.num_hidden_layers + 1)
        ]
        self.embeddings = Dinov2Embeddings(config)
        self.encoder = Dinov2Encoder(config)

        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # Initialize weights and apply final processing
        # self.post_init()

    def get_input_embeddings(self) -> Dinov2PatchEmbeddings:
        return self.embeddings.patch_embeddings

    def forward(
        self,
        pixel_values: torch.Tensor,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> BackboneOutput:
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )

        embedding_output = self.embeddings(pixel_values)

        outputs = self.encoder(
            embedding_output,
            output_hidden_states=True,
            output_attentions=output_attentions,
            return_dict=return_dict,
        )

        hidden_states = outputs.hidden_states if return_dict else outputs[1]

        feature_maps = ()
        for stage, hidden_state in zip(self.stage_names, hidden_states):
            if stage in self.out_features:
                if self.config.apply_layernorm:
                    hidden_state = self.layernorm(hidden_state)
                if self.config.reshape_hidden_states:
                    hidden_state = hidden_state[:, 1:]
                    # this was actually a bug in the original implementation that we copied here,
                    # cause normally the order is height, width
                    batch_size, _, height, width = pixel_values.shape
                    patch_size = self.config.patch_size
                    hidden_state = hidden_state.reshape(
                        batch_size, height // patch_size, width // patch_size, -1
                    )
                    hidden_state = hidden_state.permute(0, 3, 1, 2).contiguous()
                feature_maps += (hidden_state,)

        if not return_dict:
            if output_hidden_states:
                output = (feature_maps,) + outputs[1:]
            else:
                output = (feature_maps,) + outputs[2:]
            return output

        return BackboneOutput(
            feature_maps=feature_maps,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=outputs.attentions if output_attentions else None,
        )

    def forward_with_filtered_kwargs(self, *args, **kwargs):
        signature = dict(inspect.signature(self.forward).parameters)
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in signature}
        return self(*args, **filtered_kwargs)


def load_backbone(config):
    backbone_config = getattr(config, "backbone_config", None)
    backbone = Dinov2Backbone(config=backbone_config)
    return backbone


class DepthAnythingForDepthEstimation(DepthAnythingPreTrainedModel):
    config_class = DepthAnythingConfig

    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.backbone = load_backbone(config)
        self.neck = DepthAnythingNeck(config)
        self.head = DepthAnythingDepthEstimationHead(config)

        # Initialize weights and apply final processing
        # self.post_init()

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple[torch.Tensor], DepthEstimatorOutput]:
        loss = None
        if labels is not None:
            raise NotImplementedError("Training is not implemented yet")

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )

        outputs = self.backbone.forward_with_filtered_kwargs(
            pixel_values,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
        )
        hidden_states = outputs.feature_maps

        _, _, height, width = pixel_values.shape
        patch_size = self.config.patch_size
        patch_height = height // patch_size
        patch_width = width // patch_size

        hidden_states = self.neck(hidden_states, patch_height, patch_width)

        predicted_depth = self.head(hidden_states, patch_height, patch_width)

        if not return_dict:
            if output_hidden_states:
                output = (predicted_depth,) + outputs[1:]
            else:
                output = (predicted_depth,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return DepthEstimatorOutput(
            loss=loss,
            predicted_depth=predicted_depth,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=outputs.attentions,
        )


def _dict_from_json_file(json_file: Union[str, os.PathLike]):
    with open(json_file, encoding="utf-8") as reader:
        text = reader.read()
    import json

    return json.loads(text)


def check():
    model = DepthAnythingForDepthEstimation.from_pretrained(
        "depth-anything/Depth-Anything-V2-Small-hf"
    )
    # compare one layer
    weight = model.state_dict()[
        "backbone.encoder.layer.0.attention.attention.key.weight"
    ]
    logging.info(weight.dtype)
    logging.info(weight.shape)
    logging.info(weight[0][0:10])

    from transformers import DPTImageProcessor

    image_processor = DPTImageProcessor(
        do_resize=True,
        size={"height": 518, "width": 518},
        ensure_multiple_of=14,
        keep_aspect_ratio=True,
        do_rescale=True,
        do_normalize=True,
        image_mean=[0.485, 0.456, 0.406],
        image_std=[0.229, 0.224, 0.225],
    )

    # import transformers
    #
    # image_processor = transformers.AutoImageProcessor.from_pretrained(
    #     "depth-anything/Depth-Anything-V2-Small-hf"
    # )

    image = Image.open("images/room.jpg")
    inputs = image_processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    post_processed_output = image_processor.post_process_depth_estimation(
        outputs,
        target_sizes=[(image.height, image.width)],
    )

    predicted_depth = post_processed_output[0]["predicted_depth"]
    depth = (predicted_depth - predicted_depth.min()) / (
        predicted_depth.max() - predicted_depth.min()
    )
    depth = depth.detach().cpu().numpy() * 255
    depth = Image.fromarray(depth.astype("uint8"))
    depth.show()

    # model = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")


#    sd_hf = model.state_dict()
#    for k, v in sd_hf.items():
#        print (k, v.shape)


def act():
    print("act")
    return True


import sys
import unittest


class TestScript(unittest.TestCase):
    def setUp(self):
        self._old_stdout = sys.stdout
        sys.stdout = io.TextIOWrapper(io.BytesIO(), sys.stdout.encoding)

    def test_foo(self):
        print(act())
        sys.stdout.seek(0)
        self.assertEqual(sys.stdout.read(), "act\nTrue\n")

    def tearDown(self):
        sys.stdout.close()
        sys.stdout = self._old_stdout


if __name__ == "__main__":
    check()
