# Projeto Classificador de Frases com Base em Sentimentos

## Descrição
Este projeto implementa um classificador de frases de acordo com sentimentos usando o algoritmo Support Vector Machine (SVM) treinado com vetorização TF-IDF.

## Requisitos
- Python 3.x
- Bibliotecas:
  - `nltk`
  - `scikit-learn`

Instale as dependências usando:
```bash
pip install nltk scikit-learn
```

Se necessário, descomente esta parte do código (para fazer download dos recursos da biblioteca NLTK):
```python
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
```

## Usage
1. Garanta que `frases_positivas.txt` e `frases_negativas.txt` estão populados de frases em Português.
2. Execute a instrução:
   ```bash
   python main.py
   ```
3. Digite uma frase no prompt e pressione Enter.

## Como Funciona

1. **Carregamento de Dados**: Lê frases positivas e negativas a partir de arquivos de texto.
2. **Pré-processamento**: Tokeniza o texto, remove stopwords em português e filtra caracteres não alfanuméricos.
3. **Extração de Características**: Converte o texto em vetores TF-IDF.
4. **Treinamento do Modelo**: Divide os dados em conjuntos de treinamento (70%) e teste (30%), e então treina um modelo SVM linear.
5. **Predição**: Classifica novas frases utilizando o modelo e o vetorizar treinados.

## Exemplo
Input:
```
Digite uma frase: Este filme é incrível!
```
Output:
```
Frase: 'Este filme é incrível!' -> Positiva
