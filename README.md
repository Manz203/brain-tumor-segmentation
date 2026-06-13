\# Brain Tumor Segmentation using U-Net with Adversarial Training



A deep learning system for automatic brain tumor segmentation from MRI scans, using a U-Net based generator trained in an adversarial (GAN-style) setup with a PatchGAN discriminator. Includes a full inference pipeline and a web-based interface for uploading scans and viewing predicted segmentation masks.



\## Overview



Brain tumor segmentation is a critical task in medical imaging, helping clinicians localize and quantify tumor regions from MRI scans. This project implements an encoder-decoder U-Net architecture as the segmentation generator, trained alongside a PatchGAN discriminator to encourage anatomically realistic mask predictions.



\## Architecture



\*\*Generator (UNetGenerator)\*\* — A 4-level U-Net with skip connections:

\- Encoder: 4 downsampling blocks (64 → 128 → 256 → 512 channels), each with two Conv-BatchNorm-ReLU layers followed by max pooling

\- Bottleneck: 1024-channel convolutional block

\- Decoder: 4 upsampling blocks using transposed convolutions, with skip connections concatenating corresponding encoder features

\- Output: 1-channel sigmoid activation producing a binary segmentation mask

\- \*\*Total trainable parameters: \~31.04M\*\*



\*\*Discriminator (PatchDiscriminator)\*\* — A PatchGAN-style convolutional discriminator that takes the concatenated image and mask (4 channels) and outputs a patch-wise real/fake prediction, encouraging locally realistic mask structure.



\## Dataset



\- Training samples: 2,799

\- Validation samples: 1,119

\- Images resized to 256×256

\- Custom train/validation split via `split\\\_dataset.py`



\## Training



Trained for 50 epochs on GPU (CUDA) with a learning rate of 1e-3.



Validation Dice score progression (selected epochs):

| Epoch | Train Dice | Val Dice |

|-------|-----------|----------|

| 1     | 0.2086    | 0.2788   |

| 5     | 0.4571    | 0.4775   |

| 10    | 0.5585    | 0.5848   |

| 12    | 0.5827    | \*\*0.6154\*\* |

| 14    | 0.5926    | \*\*0.6103\*\* |



Best validation Dice achieved: \*\*\~0.62\*\*, with the model still improving at later epochs (full 50-epoch run extends performance further).



\## Project Structure

brain-tumor-segmentation/



├── model.py              # U-Net generator and PatchGAN discriminator architectures



├── train\_unet.py         # Training script for the U-Net model



├── train.py               # Adversarial (GAN-style) training script



├── evaluate.py            # Model evaluation



├── evaluate\_unet.py        # U-Net specific evaluation with Dice/IoU metrics



├── inference.py           # Run inference on new MRI images



├── split\_dataset.py       # Train/validation dataset splitting



├── api.py                 # Backend API for serving predictions



├── app.py                 # Web application server



├── index.html             # Frontend interface



├── script.js               # Frontend logic



├── Notebooks/              # Exploratory notebooks and experiments



└── requirements.txt

\## Setup \& Usage



\### Install dependencies

```bash

pip install -r requirements.txt

```



\### Train the model

```bash

python train\_unet.py

```



\### Run inference on a new image

```bash

python inference.py --image path/to/scan.png

```



\### Run the web app

```bash

python app.py

```

Then open `index.html` in your browser to upload MRI scans and view predicted segmentation masks.



\## Model Weights



Trained model checkpoints (`best\_unet.pth`, `final\_generator.pth`, `best\_generator.pth`) are not included in this repository due to size (\~125MB each), but can be provided on request or via \[Google Drive link].



\## Future Work



\- Extend training beyond 50 epochs with learning rate scheduling

\- Experiment with additional loss functions (combined Dice + BCE + adversarial loss tuning)

\- Add multi-class segmentation for tumor sub-regions (edema, enhancing tumor, necrotic core)

\- Deploy as a hosted web demo (e.g. Hugging Face Spaces)



