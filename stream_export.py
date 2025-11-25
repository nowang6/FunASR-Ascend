#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

import argparse
import os

import torch
from omegaconf import DictConfig, ListConfig, OmegaConf

from funasr.download.download_model_from_hub import download_model
from funasr.models.paraformer_streaming.model import ParaformerStreaming
from funasr.register import tables
from funasr.train_utils.load_pretrained_model import load_pretrained_model
from funasr.train_utils.set_all_random_seed import set_all_random_seed
from funasr.utils.misc import deep_update


def _to_python(obj):
    """Convert OmegaConf containers to native python types."""
    if isinstance(obj, (DictConfig, ListConfig)):
        return OmegaConf.to_container(obj, resolve=True)
    return obj


def load_paraformer_streaming(model_path, model_revision="master", device=None):
    """Build ParaformerStreaming the same way AutoModel would."""
    kwargs = download_model(model=model_path, model_revision=model_revision)
    kwargs = _to_python(kwargs)
    set_all_random_seed(kwargs.get("seed", 0))

    if device is None:
        device = kwargs.get("device", "cuda")
    if (
        (device == "cuda" and not torch.cuda.is_available())
        or (device == "xpu" and not torch.xpu.is_available())
        or (device == "mps" and not torch.backends.mps.is_available())
        or kwargs.get("ngpu", 1) == 0
    ):
        device = "cpu"
    kwargs["device"] = device
    torch.set_num_threads(kwargs.get("ncpu", 4))

    tokenizer = kwargs.get("tokenizer", None)
    kwargs["vocab_size"] = -1
    if tokenizer is not None:
        tokenizer_names = tokenizer.split(",") if isinstance(tokenizer, str) else tokenizer
        tokenizers_conf = kwargs.get("tokenizer_conf", {})
        if not isinstance(tokenizers_conf, (list, tuple, ListConfig)):
            tokenizers_conf = [tokenizers_conf] * len(tokenizer_names)
        built_tokenizers = []
        token_lists = []
        vocab_sizes = []
        for tokenizer_name, tokenizer_conf in zip(tokenizer_names, tokenizers_conf):
            tokenizer_conf = _to_python(tokenizer_conf)
            tokenizer_class = tables.tokenizer_classes.get(tokenizer_name)
            tokenizer_inst = tokenizer_class(**tokenizer_conf)
            built_tokenizers.append(tokenizer_inst)
            token_list = tokenizer_inst.token_list if hasattr(tokenizer_inst, "token_list") else None
            if token_list is None and hasattr(tokenizer_inst, "get_vocab"):
                token_list = tokenizer_inst.get_vocab()
            token_lists.append(token_list)
            vocab_sizes.append(len(token_list) if token_list is not None else -1)
        if len(built_tokenizers) == 1:
            built_tokenizers = built_tokenizers[0]
            token_lists = token_lists[0]
            vocab_sizes = vocab_sizes[0]
        kwargs["tokenizer"] = built_tokenizers
        kwargs["token_list"] = token_lists
        kwargs["vocab_size"] = vocab_sizes

    frontend = kwargs.get("frontend", None)
    kwargs["input_size"] = None
    if frontend is not None:
        frontend_class = tables.frontend_classes.get(frontend)
        frontend_conf = _to_python(kwargs.get("frontend_conf", {}))
        frontend = frontend_class(**frontend_conf)
        kwargs["input_size"] = frontend.output_size() if hasattr(frontend, "output_size") else None
    kwargs["frontend"] = frontend

    model_conf = {}
    deep_update(model_conf, _to_python(kwargs.get("model_conf", {})))
    deep_update(model_conf, kwargs)
    if kwargs.get("model") != ParaformerStreaming.__name__:
        raise ValueError(f"Expected ParaformerStreaming model, but got {kwargs.get('model')}")
    model_class = tables.model_classes.get(kwargs["model"])
    model = model_class(**model_conf)

    init_param = kwargs.get("init_param", None)
    if init_param is not None and os.path.exists(init_param):
        load_pretrained_model(
            model=model,
            path=init_param,
            ignore_init_mismatch=kwargs.get("ignore_init_mismatch", True),
            oss_bucket=kwargs.get("oss_bucket", None),
            scope_map=kwargs.get("scope_map", []),
            excludes=kwargs.get("excludes", None),
        )
    model.to(device)
    model.eval()
    return model, kwargs


def parse_args():
    parser = argparse.ArgumentParser(description="Export ParaformerStreaming to ONNX.")
    parser.add_argument(
        "--model-path",
        default="models/speech_paraformer-large_asr",
        help="Local model directory or hub identifier.",
    )
    parser.add_argument(
        "--model-revision",
        default="v2.0.4",
        help="Model revision used by download_model.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cuda", "cpu", "xpu", "mps"],
        help="Device used to build the model before export (defaults to CPU).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to place exported ONNX files. Defaults to <model-path>/onnx.",
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Enable dynamic quantization on the exported ONNX models.",
    )
    parser.add_argument(
        "--opset-version",
        type=int,
        default=14,
        help="ONNX opset version.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging inside torch.onnx.export.",
    )
    return parser.parse_args()


def _prepare_dummy_input(dummy_input, device):
    if isinstance(dummy_input, torch.Tensor):
        return dummy_input.to(device)
    return tuple(inp.to(device) for inp in dummy_input)


def _export_single_model_to_onnx(
    model, export_dir, opset_version, verbose, device, quantize
):
    dummy_input = _prepare_dummy_input(model.export_dummy_inputs(), device)

    export_name = model.export_name + ".onnx" if isinstance(model.export_name, str) else model.export_name()
    model_path = os.path.join(export_dir, export_name)

    torch.onnx.export(
        model,
        dummy_input,
        model_path,
        verbose=verbose,
        do_constant_folding=True,
        opset_version=opset_version,
        input_names=model.export_input_names(),
        output_names=model.export_output_names(),
        dynamic_axes=model.export_dynamic_axes(),
    )

    if quantize:
        try:
            from onnxruntime.quantization import QuantType, quantize_dynamic
            import onnx
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "Quantization requested but onnxruntime is not available. "
                "Install it via `pip install onnx onnxruntime`."
            ) from exc

        quant_model_path = model_path.replace(".onnx", "_quant.onnx")
        onnx_model = onnx.load(model_path)
        nodes = [n.name for n in onnx_model.graph.node]
        nodes_to_exclude = [
            n for n in nodes if "output" in n or "bias_encoder" in n or "bias_decoder" in n
        ]
        quantize_dynamic(
            model_input=model_path,
            model_output=quant_model_path,
            op_types_to_quantize=["MatMul"],
            per_channel=True,
            reduce_range=False,
            weight_type=QuantType.QUInt8,
            nodes_to_exclude=nodes_to_exclude,
        )


def _export_models_to_onnx(model, export_kwargs):
    model_scripts = model.export(**export_kwargs)
    export_dir = export_kwargs["output_dir"]
    os.makedirs(export_dir, exist_ok=True)

    if not isinstance(model_scripts, (list, tuple)):
        model_scripts = (model_scripts,)

    for model_script in model_scripts:
        model_script.eval()
        _export_single_model_to_onnx(
            model_script,
            export_dir=export_dir,
            opset_version=export_kwargs["opset_version"],
            verbose=export_kwargs["verbose"],
            device=export_kwargs["device"],
            quantize=export_kwargs["quantize"],
        )

    return export_dir


def export_paraformer_to_onnx(model, cfg, args):
    export_kwargs = dict(cfg)
    # Remove keys that would collide with positional arguments inside export().
    export_kwargs.pop("model", None)
    export_kwargs.update(
        {
            "type": "onnx",
            "quantize": args.quantize,
            "opset_version": args.opset_version,
            "device": args.device,
            "verbose": args.verbose,
            "output_dir": args.output_dir
            or os.path.join(cfg.get("model_path", args.model_path), "onnx"),
        }
    )

    with torch.no_grad():
        export_dir = _export_models_to_onnx(model=model, export_kwargs=export_kwargs)
    return export_dir


def main():
    args = parse_args()
    model, cfg = load_paraformer_streaming(
        model_path=args.model_path, model_revision=args.model_revision, device=args.device
    )
    export_dir = export_paraformer_to_onnx(model, cfg, args)
    print(f"ONNX models exported to: {export_dir}")


if __name__ == "__main__":
    main()
