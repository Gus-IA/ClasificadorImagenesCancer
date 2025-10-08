# Clasificador de Imágenes de Cáncer de Piel (Benigno vs Maligno) 🧠🔬

Este proyecto implementa un **clasificador de imágenes médicas** para distinguir entre cáncer de piel benigno y maligno utilizando redes neuronales convolucionales (CNN) y técnicas de aprendizaje profundo con PyTorch.

---

## 📁 Dataset

El dataset utilizado se descarga automáticamente desde el siguiente enlace:

> `https://mymldatasets.s3.eu-de.cloud-object-storage.appdomain.cloud/skin-cancer-malignant-vs-benign.zip`

Contiene imágenes de dos clases:
- **benign**
- **malignant**

El conjunto se divide en entrenamiento, validación y prueba.

---

## 🧠 Modelos implementados

- 🔧 **CNN personalizada tipo VGG** (implementada desde cero)
- 🏗️ **ResNet18** desde torchvision:
  - Desde cero (sin pesos preentrenados)
  - Usando pesos preentrenados en ImageNet
  - Con fine-tuning (unfreezing)

---

## 📈 Funcionalidades

- Descarga y extracción automática del dataset
- Visualización de muestras
- Entrenamiento con early stopping
- Evaluación en conjunto de test
- Representación gráfica de métricas
- Predicción y fine-tuning
- Uso de GPU si está disponible (`cuda`)

---

## 📦 Requisitos

Instala las dependencias con:

```bash
pip install -r requirements.txt

🧑‍💻 Autor

Desarrollado por Gus como parte de su aprendizaje en Python e IA.
