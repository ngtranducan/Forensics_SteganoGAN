# **StegenoGAN fixed (2023)**
## Original SteganoGAN (https://github.com/DAI-Lab/SteganoGAN)
* License: MIT
* Documentation: (https://DAI-Lab.github.io/SteganoGAN)

## **Overview**
___
This SteganoGAN repo is fixed for new update of **torch**, **python**, **torchvision** instead of the old version from author.

I fixed code of this package for my Forensics project about Steganograpy using Machine Learning (GAN).

## **Requirements**
___
* python 3.x
* torch == 1.9.0
* torchvision == 0.10.0

## **Installation**
___
Remember to check the versions of the packages in the requirements section first.

1. Download Steganogan for python, recommend using `pip`
```bash
pip install steganogan
```

You can also build from source in author's git but I don't recommend because I can't do during project implementation.

2. Find and replace 2 files ``utils.py`` and ``models.py`` in steganogan's folder with 2 files in ``fix_2023`` folder in this github: 

* **Linux**: ``/usr/local/lib/<version>/dist-packages/pip/steganogan/``
* **Windows**: ``C:\Users\<Username>\AppData\Local\Programs\Python\ <Python_version>\Lib\site-packages\steganogan\``

The 2 files above have been edited to match the new versions of the related packages.

## **Usage**
___
### **Command line**
You can use Steganogan for encoding and decoding staganograpy images (should be `.png`)

**Hide message inside an image**

```
steganogan encode [options] <path_to_image> "Message"
```

**Decode message from image (if it has)**

```
steganogan decode [options] <path_to_image>
```

**Additional options**

The script has some additional options to control its behavior:

* `-o, --output PATH`: Path where the generated image will be stored. Defaults to `output.png`.
* `-a, --architecture ARCH`: Architecture to use, basic or dense. Defaults to dense.
* `-v, --verbose`: Be verbose.
* `--cpu`: force CPU usage even if CUDA is available. This might be needed if there is a GPU
  available in the system but the VRAM amount is too low.

**Warning**: use the same *--architecture* option when encode and decode.

### **Python**
You can use **SteganoGAN** module from python code using `steganogan.SteganoGAN` class
This class can be instantiated using a pretrained model with 2 option of model: **dense** or **basis**

```
from steganogan import SteganoGAN
steganogan = SteganoGAN.load(architecture='dense')
```
After load model, you can using ``.encode`` attributed with image path, output image path and message to encode message inside images

```
steganogan.encode('input.png', 'output.png', 'This is a super secret message!')
```

This will generate new ``output.png`` with smaller size than the original but closely resembles the original but contains the secret message.

If you want to recover message, use ``.dcode`` method:
```
steganogan.decode('output.png')
> This is a super secret message!
```
In this repo, I have attached a .ipynb file describing the simple uses of SteganoGAN

## **Research**
___
The author has provided 2 datasets for you to create your own models, DIV2K and COCO2017 in the research folder.

Also you can find other images datasets on the internet.

## **Citing SteganoGAN**
___
If you use SteganoGAN for your research, please consider citing the following work:

Zhang, Kevin Alex and Cuesta-Infante, Alfredo and Veeramachaneni, Kalyan. SteganoGAN: High
Capacity Image Steganography with GANs. MIT EECS, January 2019. ([PDF](https://arxiv.org/pdf/1901.03892.pdf))

```
@article{zhang2019steganogan,
  title={SteganoGAN: High Capacity Image Steganography with GANs},
  author={Zhang, Kevin Alex and Cuesta-Infante, Alfredo and Veeramachaneni, Kalyan},
  journal={arXiv preprint arXiv:1901.03892},
  year={2019},
  url={https://arxiv.org/abs/1901.03892}
}
```




