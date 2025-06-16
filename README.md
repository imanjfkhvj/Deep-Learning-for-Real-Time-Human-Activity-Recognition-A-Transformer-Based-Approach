# Deep Learning for Real-Time Human Activity Recognition – A Transformer-Based Approach

## A Transformer-based model that recognizes human activities from video sequences in real time.

This project implements a deep learning model based on the Transformer architecture to perform real-time human activity recognition (HAR) from video inputs. By capturing both spatial and temporal features, the model aims to classify human actions accurately and efficiently. The approach improves upon traditional CNN-LSTM models, offering better scalability and performance for sequential video data.

![HAR process(Approach by Transformers)]([C:\Users\hp\Desktop\HAR\harimg](https://www.google.com/url?sa=i&url=https%3A%2F%2Fwww.mdpi.com%2F1424-8220%2F25%2F2%2F301&psig=AOvVaw0sC5d6PrPtgTGaNyoMYB1-&ust=1750197296561000&source=images&cd=vfe&opi=89978449&ved=0CBQQjRxqFwoTCNjKw5L39o0DFQAAAAAdAAAAABAK).jpg)


---

## Features

- Real-time recognition of human activities from video input  
- Transformer-based architecture for temporal modeling  
- Custom preprocessing and dataset adaptation  
- Implementation in Python using PyTorch  
- Real-time webcam activity detection  

---

## Tools & Technologies

- Python  
- PyTorch  
- Google Colab  
- Spyder IDE  
- OpenCV  
- NumPy  

---

## Dataset

This project uses 2D video datasets adapted for real-time action recognition. While inspired by the 3D-based SpATr model, this implementation focuses on lightweight 2D sequence processing.

---

## Model Architecture

The model separates spatial and temporal processing:

- Spatial feature extraction via custom CNN layers or pretrained backbones  
- Temporal modeling using a Transformer encoder with positional encoding and multi-head self-attention  
- Fully connected layers for classification  

---

## How to Run

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Preprocess your dataset**  
   Place your video sequences or extracted frames in the `data/` folder.

3. **Train the model**
   ```bash
   python avec chaque epoch.ipy
   ```

4. **Run real-time activity recognition using webcam**
   ```bash
   python test.py 
   python vitAvecCamera.py
   ```



##  Future Work

- Optimize performance for low-resource or embedded devices  
- Extend to multimodal input (e.g., audio + video)  
- Explore ViViT and hybrid CNN-Transformer models  






