# Deep Learning for Real-Time Human Activity Recognition – A Transformer-Based Approach

## A Transformer-based model that recognizes human activities from video sequences in real time.

This project implements a deep learning model based on the Transformer architecture to perform real-time human activity recognition (HAR) from video inputs. By capturing both spatial and temporal features, the model aims to classify human actions accurately and efficiently. The approach improves upon traditional CNN-LSTM models, offering better scalability and performance for sequential video data.

> **Architecture Credit:**  
Our model is inspired by a pre-existing Transformer architecture used in a previous research project. The original implementation can be found here:  
**[SpATr: A Spatial-Temporal Transformer for HAR](https://github.com/h-bouzid/spatr/blob/main/motion_transformer/vit.py)**

![HAR process(Approach by Transformers)](https://www.mdpi.com/sensors/sensors-25-00301/article_deploy/html/images/sensors-25-00301-g001.png)

---

##  Features

- Real-time recognition of human activities from video input  
- Transformer-based architecture for temporal modeling  
- Custom preprocessing and dataset adaptation  
- Implementation in Python using PyTorch  
- Real-time webcam activity detection  

---

##  Tools & Technologies

- Python  
- PyTorch  
- Google Colab  
- Spyder IDE  
- OpenCV  
- NumPy  

---

##  Important Notes

>  **Sensitive Code**  
Some parts of our training notebook contain sensitive or private logic and have been removed from the public repository for confidentiality reasons. If you’re interested in the full implementation or further guidance, feel free to contact me at:  
 **imane@bensadik.net**

>  **Pair**  
This project was made possible thanks to my amazing project partner:  
 **@hajarCH02** <3

---

##  Dataset

We worked with adapted 2D video datasets for real-time HAR. While our method draws inspiration from 3D Transformer approaches like SpATr, our focus was on building a lightweight and fast 2D sequence solution.

---

##  Model Architecture

The model separates spatial and temporal processing:

- **Spatial features:** Extracted via lightweight CNN layers or pretrained backbones  
- **Temporal features:** Modeled using a Transformer encoder with multi-head self-attention  
- **Classification:** Final fully connected layers  

---

##  Future Work

- Optimize for low-resource and edge devices  
- Explore ViViT and hybrid CNN-Transformer models  
- Add multimodal input (e.g., audio + video)  



