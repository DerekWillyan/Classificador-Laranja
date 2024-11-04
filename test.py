import tkinter as tk
from tkinter import filedialog, Label, Button
from PIL import Image, ImageTk
import torch
from torchvision import transforms
from train import SimpleCNN  # Importe a classe SimpleCNN do arquivo do modelo

# Definindo parâmetros de imagem
img_height, img_width = 150, 150

# Transformações de imagem (mesmas do treinamento)
data_transforms = transforms.Compose([
    transforms.Resize((img_height, img_width)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Carregar o modelo salvo
model = SimpleCNN(img_height, img_width)
model.load_state_dict(torch.load('orange_classifier_pytorch.pth'))
model.eval()

# Função para classificar uma imagem
def classify_image(img_path):
    img = Image.open(img_path).convert("RGB")  # Converter para RGB caso seja necessário
    img = data_transforms(img)
    img = img.unsqueeze(0)  # Adiciona uma dimensão para o batch

    with torch.no_grad():
        output = model(img)
        prediction = "Podre" if output.item() > 0.5 else "Fresca"
        return prediction

# Função para carregar e exibir a imagem
def load_image():
    global img, img_label
    img_path = filedialog.askopenfilename()
    if img_path:
        # Classificar a imagem
        prediction = classify_image(img_path)
        
        # Carregar e exibir a imagem
        img = Image.open(img_path).resize((200, 200))  # Redimensiona para exibição na GUI
        img_tk = ImageTk.PhotoImage(img)
        img_label.config(image=img_tk)
        img_label.image = img_tk
        
        # Atualizar o texto de classificação
        result_label.config(text=f"Classificação: {prediction}")

# Configuração da GUI
root = tk.Tk()
root.title("Classificador de Laranjas")
root.geometry("400x400")

# Botão para carregar imagem
btn_load = Button(root, text="Carregar Imagem", command=load_image)
btn_load.pack(pady=20)

# Label para exibir a imagem carregada
img_label = Label(root)
img_label.pack()

# Label para exibir o resultado da classificação
result_label = Label(root, text="Classificação: ", font=("Arial", 14))
result_label.pack(pady=20)

# Iniciar o loop da GUI
root.mainloop()
