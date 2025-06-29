# 🧠 DCGAN - CelebA Face Generator

This project implements a Deep Convolutional GAN (DCGAN) to generate realistic human face images using the CelebA dataset.

---

## 📂 Project Structure
├── checkpoints/ # Model checkpoints (.pth files)

├── generated_images/ # Output images from generator

├── model.py # Generator and Discriminator definition

├── train.py # Training script

├── test.py # Testing / inference script

├── tensorboard/ # Training logs

├── requirements.txt # Required packages

└── README.md # You're reading this


---

 🚀 How to Run

1. Clone the repo

```bash
git clone https://github.com/your-username/DCGans-Celeba-Faces.git
cd DCGans-Celeba-Faces
```

2. Download dataset
```bash
https://drive.google.com/drive/folders/1iopqsfD_a_caiqj2D8F23bEUsIaDphjL?usp=sharing
```
 or download from https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html

3. Install dependencies
```bash
pip install -r requirements.txt
```
4. Train the model
```bash
python train.py
```
Default dataset: CelebA (you need to download it manually)
Adjust hyperparameters in train.py if needed.

5. Generate images
```bash
python test.py
```
This loads the pretrained model and generates fake face images saved in generated_images/.

📦 Pretrained Models
✅ dcgan_best.pth: Best checkpoint (lowest validation loss)

✅ last.pth: Last checkpoint after final epoch


https://github.com/user-attachments/assets/034d7231-42d7-4577-bbe9-11f94b246119



📊 TensorBoard
To monitor training:
```
tensorboard --logdir=tensorboard/
```

Loss D and G
![image](https://github.com/user-attachments/assets/a8a8bf9d-2c1c-44e5-a0bd-867bb1f9196d)


