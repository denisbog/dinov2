import collections
import inspect
import os
from collections import OrderedDict
from pathlib import Path
from typing import Any, Optional, Union
import torch
from PIL import Image
from safetensors import safe_open
from torch import nn

from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.activations import ACT2FN
from transformers.utils.generic import torch_int
import enum
from tqdm import tqdm
import logging
from typing import Callable

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class BackboneType(enum.Enum):
    TRANSFORMERS = "transformers"


def is_torch_available():
    return True


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
    out_indices = [stage_names.index(layer) for layer in out_features]
    return out_features, out_indices


def get_aligned_output_features_output_indices(
    out_features: Optional[list[str]],
    out_indices: Optional[Union[list[int], tuple[int]]],
    stage_names: list[str],
) -> tuple[list[str], list[int]]:
    out_indices = list(out_indices) if out_indices is not None else None
    # First verify that the out_features and out_indices are valid
    output_features, output_indices = _align_output_features_output_indices(
        out_features=out_features, out_indices=out_indices, stage_names=stage_names
    )
    # Verify that the aligned out_features and out_indices are valid
    return output_features, output_indices


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
):
    with safe_open(checkpoint_file, framework="pt") as f:
        state_dict = {}
        for k in f.keys():
            _slice = f.get_slice(k)
            k_dtype = _slice.get_dtype()
            dtype = str_to_torch_dtype[k_dtype]
            state_dict[k] = torch.empty(
                size=_slice.get_shape(), dtype=dtype, device="meta"
            )
        return state_dict


def _infer_parameter_dtype(
    model: "LocalPretrainedModel",
    param_name: str,
) -> Union[bool, Optional[torch.dtype]]:
    old_param = model.get_parameter(param_name)
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

    is_meta_state_dict = shard_file.endswith(".safetensors")
    file_pointer = None
    if is_meta_state_dict:
        file_pointer = safe_open(shard_file, framework="pt", device=tensor_device)

    for param_name, empty_param in state_dict.items():
        # we need to use serialized_param_name as file pointer is untouched
        # This is the name of the parameter as it appears on disk file
        serialized_param_name = reverse_renaming_mapping[param_name]
        param = file_pointer.get_slice(serialized_param_name)
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

        _load_parameter_into_model(model, param_name, param.to(param_device))

    if file_pointer is not None:
        file_pointer.__exit__(None, None, None)

    return None, None


def load_shard_file(args):
    (shard_file, state_dict, device_map, key_renaming_mapping, model_to_load) = args
    # If shard_file is "", we use the existing state_dict instead of loading it
    if shard_file != "":
        state_dict = load_state_dict(shard_file)

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

        # Additional attributes without default values
        for key, value in kwargs.items():
            setattr(self, key, value)

        # TODO: remove later, deprecated arguments for TF models
        self.tf_legacy_loss = kwargs.pop("tf_legacy_loss", False)
        self.use_bfloat16 = kwargs.pop("use_bfloat16", False)

    @classmethod
    def get_config_dict(
        cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs
    ) -> tuple[dict[str, Any], dict[str, Any]]:
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

    # TODO: check is needed to push the out_feature down the hierarcy
    @property
    def out_features(self):
        return self._out_features

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, os.PathLike],
        cache_dir: Optional[Union[str, os.PathLike]] = None,
        force_download: bool = False,
        local_files_only: bool = False,
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
        # Those arguments may be passed along for our internal telemetry.
        # We remove them so they don't appear in `return_unused_kwargs`.
        kwargs.pop("_from_auto", None)
        kwargs.pop("_from_pipeline", None)

        # We remove it from kwargs so that it does not appear in `return_unused_kwargs`.
        config_dict["attn_implementation"] = kwargs.pop("attn_implementation", None)

        config = cls(**config_dict)

        if hasattr(config, "pruned_heads"):
            config.pruned_heads = {
                int(key): value for key, value in config.pruned_heads.items()
            }

        logger.info(f"Model config {config}")
        return config, kwargs


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


class LocalPretrainedModel(nn.Module):
    backbone_type: Optional[BackboneType] = None

    def __init__(self, config: LocalPretrainedConfig, *inputs, **kwargs):
        super().__init__()
        self.config = config
        self.config._attn_implementation_internal = (
            self._check_and_adjust_attn_implementation(self.config._attn_implementation)
        )

    def _check_and_adjust_attn_implementation(
        self, attn_implementation: Optional[str]
    ) -> str:
        # Register kernel if relevant
        attn_implementation = self.get_correct_attn_implementation(attn_implementation)
        return attn_implementation

    def get_correct_attn_implementation(
        self, requested_attention: Optional[str]
    ) -> str:
        return requested_attention or "sdpa"

    def _init_backbone(self, config) -> None:
        self.config = config
        self.backbone_type = BackboneType.TRANSFORMERS
        self._init_transformers_backbone(config)

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

    @classmethod
    def _load_pretrained_model(
        cls,
        model,
        state_dict: Optional[dict],
        checkpoint_files: Optional[list[str]],
        pretrained_model_name_or_path: Optional[str],
        device_map: Optional[dict] = None,
    ):
        original_checkpoint_keys = list(load_state_dict(checkpoint_files[0]).keys())
        key_renaming_mapping = model._get_key_renaming_mapping(
            original_checkpoint_keys,
        )

        key_renaming_mapping = {k: v for k, v in key_renaming_mapping.items()}
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
        subfolder = kwargs.pop("subfolder", "")
        gguf_file = kwargs.pop("gguf_file", None)

        # Load config
        config_path = pretrained_model_name_or_path
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

        resolved_safetensors_file = try_to_load_from_cache(
            pretrained_model_name_or_path, "model.safetensors"
        )

        logger.info(f"loading safetensors rom file {resolved_safetensors_file}")

        checkpoint_files = [resolved_safetensors_file]

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
        self.depth_estimation_type = depth_estimation_type
        self.max_depth = max_depth if max_depth else 1


class DepthAnythingPreTrainedModel(LocalPretrainedModel):
    base_model_prefix = "depth_anything"
    main_input_name = "pixel_values"
    supports_gradient_checkpointing = True

    def _get_key_renaming_mapping(
        self,
        checkpoint_keys: list[str],
    ):
        key_renaming_mapping = {}
        for key in checkpoint_keys:
            # Class specific rename
            key_renaming_mapping[key] = key

        return key_renaming_mapping


class ModelOutput(OrderedDict):
    def __setitem__(self, key, value):
        # Will raise a KeyException if needed
        super().__setitem__(key, value)
        # Don't call self.__setattr__ to avoid recursion errors
        super().__setattr__(key, value)


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
        num_positions = self.position_embeddings.shape[1] - 1
        # always interpolate when tracing to ensure the exported model works for dynamic input shapes

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
        embeddings = self.projection(pixel_values).flatten(2).transpose(1, 2)
        return embeddings


# Copied from transformers.models.vit.modeling_vit.eager_attention_forward
def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
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
        batch_size, _, _ = hidden_states.shape
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

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return hidden_states


class Dinov2MLP(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        in_features = out_features = config.hidden_size
        hidden_features = int(config.hidden_size * config.mlp_ratio)
        self.fc1 = nn.Linear(in_features, hidden_features, bias=True)
        self.activation = ACT2FN[config.hidden_act]
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

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor]]:
        self_outputs = self.attention(hidden_states, head_mask, output_attentions)

        attention_output = self.output(self_outputs[0])

        outputs = (attention_output,) + self_outputs[
            1:
        ]  # add attentions if we output them
        return outputs


# Copied from transformers.models.beit.modeling_beit.drop_path
def drop_path(
    input: torch.Tensor, drop_prob: float = 0.0, training: bool = False
) -> torch.Tensor:
    if drop_prob == 0.0 or not training:
        return input
    keep_prob = 1 - drop_prob
    shape = (input.shape[0],) + (1,) * (
        input.ndim - 1
    )  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(
        shape, dtype=input.dtype, device=input.device
    )
    random_tensor.floor_()  # binarize
    output = input.div(keep_prob) * random_tensor
    return output


# Copied from transformers.models.beit.modeling_beit.BeitDropPath
class Dinov2DropPath(nn.Module):
    def __init__(self, drop_prob: Optional[float] = None) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return drop_path(hidden_states, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return f"p={self.drop_prob}"


class Dinov2Layer(nn.Module):
    def __init__(self, config: Dinov2Config) -> None:
        super().__init__()

        self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attention = Dinov2Attention(config)
        self.layer_scale1 = Dinov2LayerScale(config)
        self.drop_path = (
            Dinov2DropPath(
                config.drop_path_rate
            )  # drop path is not uses for evaluation but keeping it as it part of the architecture
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


class BackboneOutput(ModelOutput):
    feature_maps: Optional[tuple[torch.FloatTensor]] = None
    hidden_states: Optional[tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[tuple[torch.FloatTensor, ...]] = None


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

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

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
        )

        hidden_states = outputs.hidden_states if return_dict else outputs[1]

        feature_maps = ()
        for stage, hidden_state in zip(self.stage_names, hidden_states):
            if stage in self.out_features:
                if self.config.apply_layernorm:
                    hidden_state = self.layernorm(hidden_state)
                feature_maps += (hidden_state,)

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
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple[torch.Tensor], DepthEstimatorOutput]:
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

        return DepthEstimatorOutput(
            loss=None,
            predicted_depth=predicted_depth,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=outputs.attentions,
        )


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

    # run model on cuda
    model.to("cuda")
    ## compile is not working
    # model = torch.compile(model)
    inputs = inputs.to("cuda")

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
