import json
import math
import time
import random
import os.path
from importlib.util import find_spec
from contextlib import contextmanager
import numpy as np
import torch
from torch.nn import functional as F
from tqdm import tqdm
from stylegan.networks import Generator, Discriminator
from utilities.iter import range_batch
from utilities.math import identity, sgn, clamp, lerp
from utilities.image import save_image
from utilities.filesys import mkdirs, build_filepath
from utilities.numpy import save


class AdamSet:

    def __init__(self, alpha, beta1, beta2, conditional=False):
        self.conditional = conditional
        self.alpha = alpha
        self.betas = (beta1, beta2)

    def __iter__(self):
        for name in ("mapper", "synthesizer", "discriminator"):
            yield name, getattr(self, name + "_optimizer")
        if self.conditional:
            for name in ("generator_embedder", "discriminator_embedder", "condition_mapper"):
                yield name, getattr(self, name + "_optimizer")

    def setup(self, generator, discriminator):
        def adam(module, scale=1):
            return torch.optim.Adam(module.parameters(), lr=self.alpha * scale, betas=self.betas, eps=1e-08)

        self.mapper_optimizer = adam(generator.mapper, 0.01)
        self.synthesizer_optimizer = adam(generator.synthesizer)
        self.discriminator_optimizer = adam(discriminator.main)
        if self.conditional:
            self.generator_embedder_optimizer = adam(generator.embedder)
            self.discriminator_embedder_optimizer = adam(discriminator.embedder)
            self.condition_mapper_optimizer = adam(discriminator.mapper, 0.01)

    def update_generator(self):
        self.mapper_optimizer.step()
        self.synthesizer_optimizer.step()
        if self.conditional:
            self.generator_embedder_optimizer.step()

    def update_discriminator(self):
        self.discriminator_optimizer.step()
        if self.conditional:
            self.discriminator_embedder_optimizer.step()
            self.condition_mapper_optimizer.step()


@contextmanager
def frozen_parameters(module):
    parameters = list(module.parameters())
    trainable = [p.requires_grad for p in parameters]
    try:
        module.requires_grad_(False)
        yield
    finally:
        for parameter, enabled in zip(parameters, trainable):
            parameter.requires_grad_(enabled)


class CustomUpdater:

    def __init__(self, generator, averaged_generator, discriminator, iterator, optimizers, averaging_images=10000, lsgan=False):
        self.generator = generator.train()
        self.averaged_generator = averaged_generator.eval().requires_grad_(False)
        self.discriminator = discriminator.train()
        self.iterator = iterator
        self.optimizers = optimizers
        self.conditional = generator.conditional
        self.averaging_images = averaging_images
        self.lsgan = lsgan
        self.iteration = 0
        self.observation = {}
        self.group = None
        self.style_mixing_rate = 0.0
        self.r1_regularization_interval = 0
        self.averaged_path_length = 0.0
        self.path_length_regularization_interval = 0
        self.mode_seeking_regularization_interval = 0
        self.augumentation = False
        self.augumentation_pipeline = identity
        self.augumentation_probability = 0.0
        self.sleep_seconds = 0
        self.sleep_interval = 0

    def enable_gradient_accumulation(self, group_size):
        self.group = group_size

    def enable_style_mixing(self, mixing_rate=0.5):
        self.style_mixing_rate = mixing_rate

    def enable_r1_regularization(self, gamma=10, interval=16):
        self.r1_regularization_gamma = gamma
        self.r1_regularization_interval = interval

    def enable_path_length_regularization(self, decay=0.99, weight=2, interval=8):
        self.path_length_decay = decay
        self.path_length_penalty_weight = weight
        self.path_length_regularization_interval = interval

    def enable_mode_seeking_regularization(self, coefficient=0.1, interval=2):
        self.mode_seeking_regularization_lambda = coefficient
        self.mode_seeking_regularization_interval = interval

    def enable_adaptive_augumentation(self, pipeline, target=0.6, limit=0.8, delta_images=500000, initial_probability=None):
        self.augumentation = True
        self.augumentation_pipeline = pipeline
        self.overfitting_target = target
        self.augumentation_limit = limit
        self.augumentation_delta_images = delta_images
        if initial_probability is not None:
            self.augumentation_probability = initial_probability

    def schedule_sleep(self, seconds=1, interval=100):
        self.sleep_seconds = seconds
        self.sleep_interval = interval

    def freeze(self, generator_level, discriminator_level):
        self.generator.freeze(list(range(generator_level, self.generator.levels + 1)))
        self.discriminator.freeze(list(range(0, discriminator_level + 1)))

    def generate_latents(self, n):
        return self.generator.generate_latents(n)

    def generate_conditions(self, n, dual=False):
        if dual:
            if n % 2:
                raise ValueError("Mode seeking requires even partial batches")
            _, c = self.generator.generate_conditions(n // 2)
            return torch.cat((c, c), dim=0)
        return self.generator.generate_conditions(n)[1]

    def next_latent_groups(self, mode_seeking=False):
        for _, n in range_batch(self.batch_size, self.group_size):
            if mode_seeking and n % 2:
                raise ValueError("Mode seeking requires even partial batches")
            yield self.generate_latents(n), (self.generate_conditions(n, dual=mode_seeking) if self.conditional else None)

    def next_real_groups(self):
        batch = self.iterator.next()
        x, c = batch if self.conditional else (batch, None)
        parameter = next(self.discriminator.parameters())
        for i, n in range_batch(self.batch_size, self.group_size):
            yield x[i : i + n].to(parameter), None if c is None else c[i : i + n].to(parameter)

    def update(self):
        self.observation = {}
        self.update_core()
        self.iteration += 1
        return self.observation

    def update_core(self):
        if self.augumentation:
            self.apply_augumentation()
        # Flow through D to G without accumulating gradients of D's weights.
        with frozen_parameters(self.discriminator):
            self.update_generator()
        self.average_generator()
        rt = self.update_discriminator()
        if self.augumentation:
            self.adapt_augumentation(rt)
        self.sleep()

    def update_generator(self):
        accumulated_loss = accumulated_penalty = accumulated_path_length = accumulated_distance_ratio = 0.0
        self.generator.zero_grad(set_to_none=True)
        for z, c in self.next_latent_groups(self.mode_seeking_regularization()):
            group_size = z.shape[0]
            mix = self.generate_latents(group_size) if random.random() < self.style_mixing_rate else None
            ws, x_fake = self.generator(z, c, random_mix=mix)
            y_fake = self.discriminator(self.augumentation_pipeline(x_fake), c)
            penalty = 0.0
            if self.path_length_regularization():
                path_length = self.path_length(ws, x_fake, self.generator.generate_masks(group_size))
                averaged_path_length = lerp(path_length.mean().item(), self.averaged_path_length, self.path_length_decay)
                weight = self.path_length_penalty_weight * self.path_length_regularization_interval
                penalty = weight * ((path_length - averaged_path_length) ** 2).sum() / self.batch_size
                accumulated_path_length += path_length.sum().item() / self.batch_size
            if self.mode_seeking_regularization():
                z1, z2 = z.chunk(2, dim=0)
                x1, x2 = x_fake.chunk(2, dim=0)
                lms = self.distance_ratio(z1, x1, z2, x2).sum() / (self.batch_size / 2)
                weight = self.mode_seeking_regularization_lambda * self.mode_seeking_regularization_interval
                penalty = penalty + weight / (lms + 1e-8)
                accumulated_distance_ratio += lms.item()
            accumulated_penalty += penalty.item() if isinstance(penalty, torch.Tensor) else penalty
            loss = (self.generator_least_squares_loss(y_fake) if self.lsgan else self.generator_logistic_loss(y_fake)) / self.batch_size
            accumulated_loss += loss.item()
            (loss + penalty).backward()
        self.optimizers.update_generator()
        self.observation.update({"loss (G)": accumulated_loss, "penalty (G)": accumulated_penalty})
        if self.path_length_regularization():
            self.averaged_path_length = lerp(accumulated_path_length, self.averaged_path_length, self.path_length_decay)
            self.observation["path length"] = self.averaged_path_length
        if self.mode_seeking_regularization():
            self.observation["distance ratio"] = accumulated_distance_ratio

    @torch.no_grad()
    def average_generator(self):
        decay = 0.5 ** (self.batch_size / self.averaging_images)
        for raw, averaged in zip(self.generator.parameters(), self.averaged_generator.parameters()):
            averaged.lerp_(raw, 1 - decay)

    def update_discriminator(self):
        accumulated_loss = accumulated_penalty = accumulated_rt = 0.0
        self.discriminator.zero_grad(set_to_none=True)
        for (x_real, c_real), (z, c) in zip(self.next_real_groups(), self.next_latent_groups()):
            x_real.requires_grad_(bool(self.r1_regularization()))
            y_real = self.discriminator(self.augumentation_pipeline(x_real), c_real)
            accumulated_rt += torch.sign(y_real - 0.5 if self.lsgan else y_real).sum().item() / self.batch_size
            with torch.no_grad():
                mix = self.generate_latents(z.shape[0]) if random.random() < self.style_mixing_rate else None
                _, x_fake = self.generator(z, c, random_mix=mix)
            y_fake = self.discriminator(self.augumentation_pipeline(x_fake), c)
            penalty = 0.0
            if self.r1_regularization():
                weight = self.r1_regularization_interval * self.r1_regularization_gamma
                penalty = weight * self.gradient_penalty(x_real, y_real) / self.batch_size
                accumulated_penalty += penalty.item()
            loss = (self.discriminator_least_squares_loss(y_real, y_fake) if self.lsgan else self.discriminator_logistic_loss(y_real, y_fake)) / self.batch_size
            accumulated_loss += loss.item()
            (loss + penalty).backward()
        self.optimizers.update_discriminator()
        self.observation.update({"loss (D)": accumulated_loss, "penalty (D)": accumulated_penalty, "overfitting": accumulated_rt})
        return accumulated_rt

    def adapt_augumentation(self, overfitting):
        delta = self.batch_size / self.augumentation_delta_images
        probability = self.augumentation_probability + delta * sgn(overfitting - self.overfitting_target)
        self.augumentation_probability = clamp(0.0, probability, self.augumentation_limit)
        self.observation["augumentation"] = self.augumentation_probability

    def apply_augumentation(self):
        self.augumentation_pipeline.probability = self.augumentation_probability

    def r1_regularization(self):
        return self.r1_regularization_interval and self.iteration % self.r1_regularization_interval == 0

    def path_length_regularization(self):
        return self.path_length_regularization_interval and self.iteration % self.path_length_regularization_interval == 0

    def mode_seeking_regularization(self):
        return self.mode_seeking_regularization_interval and self.iteration % self.mode_seeking_regularization_interval == 0

    def sleep(self):
        if self.sleep_interval and self.iteration % self.sleep_interval == 0:
            time.sleep(self.sleep_seconds)

    def transfer(self, filepath, generator_level, discriminator_level):
        source_generator, source_discriminator = self.reconstruct_models(filepath, load_weights=True)
        self.generator.transfer(source_generator, list(range(generator_level, self.generator.levels + 1)))
        self.averaged_generator.transfer(source_generator, list(range(generator_level, self.averaged_generator.levels + 1)))
        self.discriminator.transfer(source_discriminator, list(range(0, discriminator_level + 1)))

    @staticmethod
    def read_snapshot(filepath):
        if os.path.splitext(filepath)[1].lower() in (".h5", ".hdf5"):
            raise ValueError("Chainer HDF5 training snapshots are unsupported; use a PyTorch .pt training snapshot")
        state = torch.load(filepath, map_location="cpu", weights_only=True)
        if state.get("kind") != "stylegan-training" or state.get("format_version") != 1:
            raise ValueError("Expected a PyTorch training snapshot, not a standalone generator")
        return state

    def load_states(self, filepath):
        state = self.read_snapshot(filepath)
        params = self.generator.model_params()
        if any(state["params"][key] != value for key, value in params.items() if key != "labels"):
            raise ValueError("Snapshot network architecture does not match")
        self.iterator.load_state_dict(state["iterator"])
        self.generator.load_state_dict(state["generator"])
        self.averaged_generator.load_state_dict(state["averaged_generator"])
        self.discriminator.load_state_dict(state["discriminator"])
        for key, optimizer in self.optimizers:
            optimizer.load_state_dict(state["optimizers"][key])
        self.averaged_path_length = state["averaged_path_length"]
        self.augumentation_probability = state["augumentation_probability"]
        self.iteration = state["iteration"]
        random.setstate(state["random"])
        name, keys, position, has_gauss, cached_gaussian = state["numpy_random"]
        np.random.set_state((name, np.array(keys, dtype=np.uint32), position, has_gauss, cached_gaussian))
        torch.set_rng_state(state["torch_random"])
        if state["cuda_random"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["cuda_random"])
        if state.get("mps_random") is not None and next(self.generator.parameters()).device.type == "mps":
            torch.mps.set_rng_state(state["mps_random"])

    def save_states(self, filepath):
        name, keys, position, has_gauss, cached_gaussian = np.random.get_state()
        torch.save(
            {
                "kind": "stylegan-training",
                "format_version": 1,
                "params": self.generator.model_params(),
                "group_size": self.discriminator.main[-1].mbstd.group_size,
                "generator": self.generator.state_dict(),
                "averaged_generator": self.averaged_generator.state_dict(),
                "discriminator": self.discriminator.state_dict(),
                "optimizers": {key: optimizer.state_dict() for key, optimizer in self.optimizers},
                "averaged_path_length": self.averaged_path_length,
                "augumentation_probability": self.augumentation_probability,
                "iteration": self.iteration,
                "iterator": self.iterator.state_dict(),
                "random": random.getstate(),
                "numpy_random": (name, keys.tolist(), position, has_gauss, cached_gaussian),
                "torch_random": torch.get_rng_state(),
                "cuda_random": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                "mps_random": torch.mps.get_rng_state() if next(self.generator.parameters()).device.type == "mps" else None,
            },
            filepath,
        )

    @staticmethod
    def read_params(snapshot):
        return snapshot["params"]

    @staticmethod
    def reconstruct_models(filepath, group_size=None, load_weights=False):
        state = CustomUpdater.read_snapshot(filepath)
        params = CustomUpdater.read_params(state)
        generator = Generator(**params)
        group_size = state["group_size"] if group_size is None else group_size
        discriminator = Discriminator(params["levels"], params["last_channels"], params["first_channels"], params["categories"], params["depth"], group_size=group_size)
        if load_weights:
            generator.load_state_dict(state["generator"])
            discriminator.load_state_dict(state["discriminator"])
        return generator, discriminator

    @property
    def batch_size(self):
        return self.iterator.batch_size

    @property
    def group_size(self):
        return self.batch_size if self.group is None else self.group

    @staticmethod
    def generator_logistic_loss(fake):
        return F.softplus(-fake).sum()

    @staticmethod
    def generator_least_squares_loss(fake):
        return ((fake - 1) ** 2).sum()

    @staticmethod
    def discriminator_logistic_loss(real, fake):
        return F.softplus(-real).sum() + F.softplus(fake).sum()

    @staticmethod
    def discriminator_least_squares_loss(real, fake):
        return ((real - 1) ** 2).sum() + (fake**2).sum()

    @staticmethod
    def gradient_penalty(x, y):
        gradient = torch.autograd.grad(y.sum(), x, create_graph=True)[0]
        return (gradient**2).sum() / 2

    @staticmethod
    def path_length(ws, x, mask):
        # Keep the original shared style tensors and mean across synthesis levels.
        gradients = torch.autograd.grad((x * mask).sum(), ws, create_graph=True)
        return torch.sqrt(torch.stack(gradients, dim=1).square().sum(dim=2).mean(dim=1))

    @staticmethod
    def distance_ratio(z1, x1, z2, x2):
        v1 = (x1 - x2).abs().mean(dim=(1, 2, 3))
        v2 = (z1 - z2).abs().mean(dim=1)
        return v1 / (v2 + 1e-8)


class CustomTrainer:

    def __init__(self, updater, epoch, dest, overwrite=False):
        self.updater = updater
        self.epoch = epoch
        self.out = dest
        self.overwrite = overwrite
        self.number = 16
        self.images_out = os.path.join(dest, "images")
        self.states_out = os.path.join(dest, "checkpoints")
        self.report_interval = self.save_interval = self.image_interval = 0
        self.progress = False
        self.records = []
        self.pending_reports = []
        mkdirs(self.images_out)
        mkdirs(self.states_out)

    def enable_reports(self, interval):
        if interval and find_spec("matplotlib") is None:
            raise RuntimeError("Training reports require Matplotlib; install requirements-training.txt or use --print-interval 0")
        self.report_interval = interval
        self.report_path = build_filepath(self.out, "report", "log", self.overwrite)
        self.curves_path = build_filepath(self.out, "curves", "png", self.overwrite)

    def enable_progress_bar(self, interval=1):
        self.progress = True

    def hook_state_save(self, interval):
        self.save_interval = interval

    def hook_image_generation(self, interval, number=None):
        self.image_interval = interval
        self.number = self.number if number is None else number

    def run(self):
        steps = math.ceil(self.epoch * len(self.updater.iterator.dataset) / self.updater.batch_size)
        started = time.monotonic()
        with tqdm(total=steps, disable=not self.progress, unit="iter") as bar:
            for _ in range(steps):
                observation = self.updater.update()
                if self.report_interval:
                    self.pending_reports.append(dict(observation))
                    if self.iteration % self.report_interval == 0:
                        self.write_report(time.monotonic() - started)
                # A snapshot includes the RNG use of its corresponding previews.
                if self.image_interval and self.iteration % self.image_interval == 0:
                    self.save_images(self)
                if self.save_interval and self.iteration % self.save_interval == 0:
                    self.save_snapshot(self)
                    self.save_generator(self)
                bar.update()
        if self.pending_reports:
            self.write_report(time.monotonic() - started)

    def write_report(self, elapsed):
        entries = {key for observation in self.pending_reports for key in observation}
        record = {key: float(np.mean([o[key] for o in self.pending_reports if key in o])) for key in entries}
        record.update(iteration=self.iteration, epoch=self.updater.iterator.epoch_detail, elapsed_time=elapsed)
        self.records.append(record)
        self.pending_reports.clear()
        with open(self.report_path, "w", encoding="utf-8") as fp:
            json.dump(self.records, fp, indent=2)
        tqdm.write("  ".join(f"{key}: {record[key]:.6g}" for key in ("iteration", "loss (G)", "loss (D)", "penalty (G)", "penalty (D)", "overfitting")))
        # Matplotlib is needed only for training reports, not for inference.
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import pyplot as plt

        figure, axes = plt.subplots()
        for key in ("loss (G)", "loss (D)", "penalty (G)", "penalty (D)", "overfitting"):
            axes.plot([r["iteration"] for r in self.records], [r[key] for r in self.records], label=key)
        axes.set_xlabel("iteration")
        axes.legend()
        figure.savefig(self.curves_path)
        plt.close(figure)

    @property
    def iteration(self):
        return self.updater.iteration

    @property
    def batch_size(self):
        return self.updater.group_size

    @property
    def conditional(self):
        return self.updater.conditional

    @staticmethod
    def save_snapshot(trainer):
        filepath = build_filepath(trainer.states_out, f"snapshot-{trainer.iteration}", "pt", trainer.overwrite)
        trainer.updater.save_states(filepath)

    @staticmethod
    def save_generator(trainer):
        filepath = build_filepath(trainer.states_out, f"generator-{trainer.iteration}", "pt", trainer.overwrite)
        trainer.updater.averaged_generator.save(filepath)

    @staticmethod
    @torch.inference_mode()
    def save_images(trainer):
        generator = trainer.updater.averaged_generator
        for i, n in range_batch(trainer.number, trainer.batch_size):
            z = generator.generate_latents(n)
            c = generator.generate_conditions(n)[1] if trainer.conditional else None
            _, y = generator(z, c)
            for j in range(n):
                filename = f"{trainer.iteration}-{i + j + 1}"
                save(build_filepath(trainer.images_out, filename + "-latent", "npy", trainer.overwrite), z[j].cpu().numpy())
                save_image(y[j].cpu().numpy(), build_filepath(trainer.images_out, filename, "png", trainer.overwrite))
