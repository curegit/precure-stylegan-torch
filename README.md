# Precure StyleGAN ADA (PyTorch ver.)

StyleGAN 2.0 implementation with Adaptive Discriminator Augmentation

This project is a PyTorch port of the [original Chainer implementation](https://github.com/curegit/precure-stylegan-ada) for continued maintenance.

## Abstract

This project follows on from [the previous project: Precure StyleGAN](https://github.com/curegit/precure-stylegan).
We aimed to generate facial images of a specific Precure (Japanese Anime) character using the StyleGAN 2.0.
We employed Adaptive Discriminator Augmentation (ADA) to improve the image quality, as the previous project showed that the dataset was too small to train decent GANs naively.
We trained the models on facial images of Cure Beauty (Smile Pretty Cure!, 2012), whose dataset size is about 3k, and achieved a score of 17.39 in FID, much better than the previous project.
We also trained the models on other common datasets, demonstrating the stability and robustness of our implementation.

![Cure Beauty](examples/beauty-curated.png)

## Requirements

- Python >= 3.14
- PyTorch >= 2.6
- Pillow >= 9.1
- NumPy >= 2.3
- tqdm

Install the inference dependencies with:

```sh
python3 -m pip install -r requirements.txt
```

Inference runs on CPU by default.
Use `-v cuda` for an available CUDA device or `-v mps` for an available MPS device, with a matching PyTorch build.
Model inspection with `show.py` always runs on CPU.

Check the available backends and PyTorch environment with:

```sh
python3 tools/inspect_env.py
```

h5py is an optional dependency needed only when converting legacy HDF5 (Chainer checkpoint) models.

## Model Zoo

Ready-to-use PyTorch generators are provided in `models/`:

| Dataset | Checkpoint | Conditioning |
| --- | --- | --- |
| Flickr-Faces-HQ | `models/ffhq.pt` | Unconditional |
| Animal Faces-HQ | `models/afhq.pt` | Conditional |
| Anime Faces | `models/anime.pt` | Unconditional |
| MNIST | `models/mnist.pt` | Conditional |
| Kuzushiji-49 | `models/k49.pt` | Conditional |

For example, inspect the MNIST model and generate images of the digit 7:

```sh
python3 show.py models/mnist.pt
python3 generate.py models/mnist.pt -n 10 -l seven -o output-mnist
```

## Results

Note that scores and example images below come from the original Chainer implementation.
These have not been re-evaluated for this PyTorch port.

We use ψ = 1.0 (no truncation applied) to evaluate for each Fréchet Inception Distance (FID).
The result grids below are uncurated, and truncation was applied with the `-C` option (shrinking to each class).

### Cure Beauty v1 (ψ = 0.8, FID = 21.98, ADA enabled, batch = 16, R1 = 100)

![Cure Beauty v1](examples/beauty-1.png)

### Cure Beauty v2 (ψ = 0.8, FID = 17.39, ADA enabled, batch = 12, R1 = 75)

The smaller batch size and lesser gradient penalty brought better quality and diversity.

![Cure Beauty v2](examples/beauty-2.png)

### Flickr-Faces-HQ (ψ = 0.9, FID = 15.61)

![FFHQ](examples/ffhq.png)

### Animal Faces-HQ (ψ = 0.8, FID = 7.64, conditional, ADA enabled)

![AFHQ](examples/afhq.png)

### Anime Faces (ψ = 0.7, FID = 13.81)

![Anime](examples/anime.png)

### MNIST (ψ = 1.1, FID = 2.61, conditional)

![MNIST](examples/mnist.png)

### Kuzushiji-49 (ψ = 1.0, FID = 3.77, conditional, ADA enabled, MSGAN)

For this training, we used Mode Seeking Regularization (MSGAN) due to problems associated with multiclass data that has a large number of categories.

![Kuzushiji-49](examples/k49.png)

## Manipulation Examples

### Style Interpolation

|          Original 1           |                  Averaged                   |          Original 2           |
| :---------------------------: | :-----------------------------------------: | :---------------------------: |
| ![Cat 1](examples/cats/1.png) | ![Cat Averaged](examples/cats/averaged.png) | ![Cat 2](examples/cats/2.png) |

### Style Mixing

#### 2-Mix

|                 Level 1–4                  |                 Level 5–7                  |                   Result                   |
| :----------------------------------------: | :----------------------------------------: | :----------------------------------------: |
| ![Level 1–4 Input](examples/mix/2/1-4.png) | ![Level 5–7 Input](examples/mix/2/5-7.png) | ![2-Mix Result](examples/mix/2/result.png) |

#### 3-Mix

|                 Level 1–3                  |                 Level 4–5                  |                 Level 6–7                  |                   Result                   |
| :----------------------------------------: | :----------------------------------------: | :----------------------------------------: | :----------------------------------------: |
| ![Level 1–3 Input](examples/mix/3/1-3.png) | ![Level 4–5 Input](examples/mix/3/4-5.png) | ![Level 6–7 Input](examples/mix/3/6-7.png) | ![3-Mix Result](examples/mix/3/result.png) |

### Interpolation Animation

![Cure Beauty v2 Animation](examples/beauty-animation.png)

## Scripts

### `show.py`

This script shows the architecture, total parameter count, CNN channels at each level, and class labels of a trained PyTorch generator checkpoint (`.pt`).

For example, to check the MNIST model, use:

```sh
python3 show.py models/mnist.pt
```

### `generate.py`

This script generates images using a trained `.pt` model.
It saves each image together with its latent vector (`N-latent.npy`) and style vector (`N-style.npy`).
Use `show.py` to inspect the available class labels.
Use the `-h` option for more details.

For example, to generate 100 dog images in the `output` folder using the AFHQ model, type:

```sh
python3 generate.py models/afhq.pt -n 100 -l dog -o output
```

`Generator.save(path)` writes a PyTorch checkpoint containing `format_version`, `params` (architecture and labels), and `state_dict` (learned tensors). `Generator.load(path, device="cpu")` reconstructs the model for inference without Chainer or h5py. Latent and style vectors continue to use `.npy` files.

`--fixed 0` fixes the per-layer injected noise.
It does not seed the randomly sampled latent vectors, so separate runs can still generate different images.

Latent and style vectors exported by the original Chainer implementation can be used with the corresponding converted Generator.
Use vectors from the same Generator checkpoint; matching vector dimensions alone do not make different models' style spaces compatible.

### `combine.py`

This script performs linear combination on style vectors from style files, creating a new style file and images.
Omitting `--coefs` averages the input styles.
It accepts PyTorch `.pt` checkpoints and existing `.npy` style vectors, including those exported by the original Chainer implementation.
Each vector must have the model's style dimension.
The result is saved as `new-style.npy`; use `-n 0` to save only the style vector.
Use the `-h` option for more details.

Example 1: To average two cats:

```sh
python3 generate.py models/afhq.pt -n 2 -l cat -t 0.8 -o cats
python3 combine.py models/afhq.pt -n 1 cats/1-style.npy cats/2-style.npy -c 0.5 0.5 -o new-cat
```

Example 2: To generate 3 human face images and add the style vector difference from image 2 to image 3 to image 1:

```sh
python3 generate.py models/ffhq.pt -n 3 -t 0.8 -o faces
python3 combine.py models/ffhq.pt -n 1 faces/1-style.npy faces/2-style.npy faces/3-style.npy -c 1 -1 1 -o new-face
```

### `mix.py`

This script mixes styles from style files and creates style-mixed images.
It accepts PyTorch `.pt` checkpoints and existing `.npy` style vectors, including those exported by the original Chainer implementation.
`...` repeats the preceding style (or the first supplied style for leading placeholders); omitted trailing levels repeat the last style.
`--lerp` interpolates between supplied styles instead, and `--justify` distributes the supplied arguments across the model's levels.
Use the `-h` option for additional details.

For instance, to generate 2 anime face images and mix them, use:

```sh
python3 generate.py models/anime.pt -n 2 -o anime
python3 mix.py models/anime.pt -n 1 anime/1-style.npy ... ... anime/2-style.npy -o anime-mix
```

### `animate.py`

This script makes an animation of the analogy using a trained generator.
It accepts PyTorch `.pt` checkpoints and writes an animated PNG (`analogy.png`), with optional GIF (`--gif`) and WebP (`--webp`) output.
Use `--prepend` and `--append` with existing `.npy` style vectors, including Chainer exports, and `-n 0` to animate only those styles.
`--loop` interpolates back to the first style, while `--repeat` sets the output animation's repeat flag.
`--frames` also saves individual frame images.
Use the `-h` option for more details.

For example, to create an animation that interpolates 10 face samples using the FFHQ model, use:

```sh
python3 animate.py models/ffhq.pt -n 10 -L -o analogy
```

To animate between two saved styles with fixed noise and additional GIF output:

```sh
python3 animate.py models/ffhq.pt -n 0 -P faces/1-style.npy -A faces/2-style.npy -I 15 --fixed 0 -L -R -G -o style-animation
```

### `train.py`

Use this script to train your own models.
Use the `-h` option for more details.

Install the training dependencies (including Matplotlib for training curves):

```sh
python3 -m pip install -r requirements-training.txt
```

Basically, you need to specify the directories for your dataset, the network architecture parameters, and the training duration.
Image data should be in RGB or Gray formats with 8-bit depth, and in powers of two square images.
Use the `-x` option to fit the network shape to the image size of the target dataset.

`--ada` switch enables Adaptive Discriminator Augmentation.

For example, to start training the AFHQ model:

```sh
# Run 100 epochs on GPU
python3 train.py -e 100 -o training-result -x 7 -b 32 -g 4 -k 8 --ada CAT_IMAGE_DIR DOG_IMAGE_DIR WILD_ANIMAL_IMAGE_DIR -l cat dog wild -v GPU
```

To resume training from a snapshot:

```sh
# Run another 50 epochs on GPU
python3 train.py -e 50 -s training-result/snapshot.pt -o training-result-resumed -b 32 -g 4 -k 8 --ada CAT_IMAGE_DIR DOG_IMAGE_DIR WILD_ANIMAL_IMAGE_DIR -l cat dog wild -v GPU
```

Snapshots restore network weights, Adam states (including learning rates and betas), EMA, iteration, path length/ADA statistics, data ordering and random states.
Pass the same dataset directories in the same order and keep the batch size unchanged.
Other training settings (regularization, accumulation/group sizes, ADA, freezing and reporting) must be passed again.
Specify the additional training length with `-e`, not the total duration.
Intermediate generators and training snapshots are saved as `.pt` files in `checkpoints/`.

Training curves at the beginning of the training typically look like this:

![Training Curve Example](examples/curves.png)

#### Transfer Learning

To perform transfer learning and fine-tuning, you can use a combination of the `-t` option and the `-Z` option.

Here are some examples of Cure Twinkle transferred from Cure Beauty v2, which was trained on approximately 300 images.

![Cure Twinkle (Transferred)](examples/transfer.png)

## Legacy HDF5 Models

The original Chainer Generator files are retained in `models/legacy/`.
To convert one, install the conversion dependencies and choose an output path:

```sh
python3 -m pip install -r tools/requirements-convert.txt
python3 tools/convert_hdf5.py models/legacy/mnist.hdf5 -o converted/mnist.pt
python3 show.py converted/mnist.pt
```

The converter requires h5py, but does not import Chainer.
It preserves architecture, labels, and learned tensors, and rejects missing, extra, or incompatible weights.
Only standalone Generator files are supported; training snapshots cannot be converted or resumed with the inference tools.

## See Also

- [StyleGAN FastAPI](https://github.com/curegit/stylegan-fastapi)
- [StyleGAN Vue](https://github.com/curegit/stylegan-vue)
- [StyleGAN Web](https://github.com/curegit/stylegan-web)
- [Precure StyleGAN](https://github.com/curegit/precure-stylegan) (old project)

## References

### Papers

- [A Style-Based Generator Architecture for Generative Adversarial Networks](https://arxiv.org/abs/1812.04948)
- [Analyzing and Improving the Image Quality of StyleGAN](https://arxiv.org/abs/1912.04958)
- [Training Generative Adversarial Networks with Limited Data](https://arxiv.org/abs/2006.06676)
- [Mode Seeking Generative Adversarial Networks for Diverse Image Synthesis](https://arxiv.org/abs/1903.05628)
- [On Leveraging Pretrained GANs for Generation with Limited Data](https://arxiv.org/abs/2002.11810)

### Datasets

- [THE MNIST DATABASE of handwritten digits](http://yann.lecun.com/exdb/mnist/)
- [Deep Learning for Classical Japanese Literature](https://arxiv.org/abs/1812.01718)
- [Anime-Face-Dataset](https://github.com/Mckinsey666/Anime-Face-Dataset)
- [Flickr-Faces-HQ Dataset (FFHQ)](https://github.com/NVlabs/ffhq-dataset)

## License

[CC BY-NC 4.0](LICENSE)
