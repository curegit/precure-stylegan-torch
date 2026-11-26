#!/usr/bin/env python3

import sys
import torch
from argparse import ArgumentParser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stylegan.networks import Generator


def hdf5_parameter_name(name, depth):
    parts = name.split(".")
    if parts[:2] == ["mapper", "mlp"]:
        # Legacy Chainer Sequential.insert registered every later dense link
        # just after the first one. Execution order differs from saved names:
        # depth=8 -> 0, 7, 6, 5, 4, 3, 2, 1. Activations have no saved weights.
        index = int(parts[2]) // 2
        parts[2] = str(0 if index == 0 else depth - index)
    return "/".join(parts)


# Read a legacy standalone Generator without importing Chainer
def load_hdf5_generator(filepath):
    import h5py

    with h5py.File(filepath, "r") as hdf5:
        names = ("size", "depth", "levels", "first_channels", "last_channels", "categories")
        missing_attrs = sorted(set((*names, "labels")) - set(hdf5.attrs))
        if missing_attrs:
            raise ValueError(f"Missing generator attributes: {', '.join(missing_attrs)}")
        params = {name: int(hdf5.attrs[name]) for name in names}
        if any(value < 1 for value in params.values()):
            raise ValueError("Generator architecture parameters must be positive")
        params["labels"] = [label.decode("utf-8") if isinstance(label, bytes) else str(label) for label in hdf5.attrs["labels"]]
        if len(params["labels"]) != params["categories"]:
            raise ValueError("Label count does not match the number of categories")
        generator = Generator(**params)
        state = generator.state_dict()
        datasets = {}

        def collect(name, obj):
            if isinstance(obj, h5py.Dataset):
                datasets[name] = obj

        hdf5.visititems(collect)
        expected_names = {hdf5_parameter_name(name, generator.depth) for name in state}
        missing = sorted(expected_names - datasets.keys())
        unexpected = sorted(datasets.keys() - expected_names)
        if missing or unexpected:
            raise ValueError(f"Generator weights do not match: missing={missing}, unexpected={unexpected}")
        for name, expected in state.items():
            source_name = hdf5_parameter_name(name, generator.depth)
            tensor = torch.from_numpy(datasets[source_name][()])
            if tensor.shape != expected.shape:
                raise ValueError(f"Shape mismatch for {source_name}: expected {tuple(expected.shape)}, got {tuple(tensor.shape)}")
            if tensor.dtype != expected.dtype:
                raise ValueError(f"Dtype mismatch for {source_name}: expected {expected.dtype}, got {tensor.dtype}")
            state[name] = tensor
        generator.load_state_dict(state)
    return generator.eval()


def convert(source, destination, overwrite=False):
    source, destination = Path(source), Path(destination)
    if source.resolve() == destination.resolve():
        raise ValueError("Input and output paths must be different")
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {destination}; use --force to overwrite")
    generator = load_hdf5_generator(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb" if overwrite else "xb") as fp:
        generator.save(fp)
    return generator


def main():
    parser = ArgumentParser(description="Convert a standalone Chainer Generator HDF5 file to a PyTorch (.pt) checkpoint")
    parser.add_argument("source", metavar="HDF5_FILE", type=Path, help="legacy generator HDF5 file (training snapshots are not supported)")
    parser.add_argument("-o", "--output", metavar="PT_FILE", type=Path, help="output checkpoint (default: source path with .pt extension)")
    parser.add_argument("-f", "--force", action="store_true", help="allow overwriting an existing output checkpoint")
    args = parser.parse_args()
    output = args.output if args.output is not None else args.source.with_suffix(".pt")
    try:
        generator = convert(args.source, output, overwrite=args.force)
    except ModuleNotFoundError as error:
        if error.name != "h5py":
            raise
        parser.exit(1, "h5py is required for conversion\n")
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"Conversion failed: {error}\n")
    print(f"Converted: {args.source} -> {output}")
    print(f"Generator: {generator.width}x{generator.height}, {generator.categories} classes, {len(generator.state_dict())} parameter tensors")


if __name__ == "__main__":
    main()
