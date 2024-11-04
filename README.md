# Classificador de Laranjas Frescas e Podres

Este projeto utiliza uma rede neural convolucional simples (CNN) implementada em PyTorch para classificar imagens de laranjas em duas categorias: frescas e podres. A aplicação inclui uma interface gráfica (GUI) construída com `tkinter` para facilitar o uso do modelo, permitindo que o usuário carregue imagens e visualize a classificação.

## Estrutura do Projeto

- **Fruit/**: Diretório contendo as imagens para treinamento e validação.
  - **FreshOrange/**: Imagens de laranjas frescas.
  - **RottenOrange/**: Imagens de laranjas podres.
- **train.py**: Script para treinar o modelo.
- **test.py**: Script que implementa a interface gráfica para classificar imagens de laranjas.

## Requisitos

- Python 3.x
- PyTorch
- torchvision
- Pillow
- tkinter

Para instalar as dependências, execute:

```bash
pip install torch torchvision pillow
```

## Treinamento do Modelo

1. Organize as imagens nas pastas `FreshOrange` e `RottenOrange`, dentro da pasta `Fruit`.
2. Execute o script `train.py` para treinar o modelo:

   ```bash
   python train.py
   ```

   O modelo treinado será salvo como `orange_classifier_pytorch.pth`.

## Uso do Classificador

Para utilizar o classificador:

1. Execute o script `test.py`:

   ```bash
   python test.py
   ```

2. Clique em "Carregar Imagem" na interface gráfica e selecione uma imagem de laranja. A classificação ("Fresca" ou "Podre") será exibida ao lado da imagem.

## Exemplo de Uso

Após carregar uma imagem, a GUI exibirá a imagem e a classificação correspondente.

![Captura de tela de 2024-11-03 21-28-56](https://github.com/user-attachments/assets/a9638fce-f825-4005-a58b-6ce28069c433)

## Personalização

- Ajuste os parâmetros do modelo, como `img_height`, `img_width`, `batch_size`, `num_epochs`, e `learning_rate` no arquivo `train.py` para modificar o treinamento conforme necessário.
- O código da GUI também pode ser modificado para incluir mais funcionalidades, como salvar os resultados.

## Contribuição

Contribuições são bem-vindas! Para relatar problemas ou sugerir melhorias, abra uma _issue_ ou envie um _pull request_.

## Licença

Este projeto é licenciado sob a [Licença MIT](LICENSE).

---

Criado por [Derek Willyan](https://github.com/DerekWillyan/)
