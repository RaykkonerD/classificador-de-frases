import nltk
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Recursos necessários do NLTK (caso ainda não tenha)
# nltk.download('punkt')
# nltk.download('punkt_tab')
# nltk.download('stopwords')

def carregar_frases(arquivo_positivo, arquivo_negativo):
    frases_positivas = []
    frases_negativas = []

    with open(arquivo_positivo, 'r', encoding='utf-8') as f:
        frases_positivas = [line.strip() for line in f.readlines()]

    with open(arquivo_negativo, 'r', encoding='utf-8') as f:
        frases_negativas = [line.strip() for line in f.readlines()]

    return frases_positivas, frases_negativas

def pre_processar_texto(texto):
    stop_words = set(stopwords.words('portuguese'))
    palavras = word_tokenize(texto.lower())
    palavras_filtradas = [palavra for palavra in palavras if palavra.isalnum() and palavra not in stop_words]
    return " ".join(palavras_filtradas)

def treinar_e_classificar(frases_positivas, frases_negativas):
    frases = frases_positivas + frases_negativas
    labels = [1] * len(frases_positivas) + [0] * len(frases_negativas)

    frases_processadas = [pre_processar_texto(frase) for frase in frases]

    # Converter as frases em vetores de características usando TF-IDF
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(frases_processadas)

    # Dividir em dados de treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.3, random_state=42)

    # Treinar o modelo SVM
    modelo_svm = SVC(kernel='linear')
    modelo_svm.fit(X_train, y_train)
    modelo_svm.predict(X_test)

    return modelo_svm, vectorizer

def classificar_frases(modelo_svm, vectorizer, nova_frase):
    nova_frase_processada = pre_processar_texto(nova_frase)
    X_nova = vectorizer.transform([nova_frase_processada])

    previsao = modelo_svm.predict(X_nova)[0]

    print(f"Frase: '{nova_frase}' -> {'Positiva' if previsao == 1 else 'Negativa'}")





# Arquivos
arquivo_positivo = 'frases_positivas.txt'
arquivo_negativo = 'frases_negativas.txt'

# Treinamento
frases_positivas, frases_negativas = carregar_frases(arquivo_positivo, arquivo_negativo)
modelo_svm, vectorizer = treinar_e_classificar(frases_positivas, frases_negativas)

# Execução
nova_frase = input("Digite uma frase: ")
classificar_frases(modelo_svm, vectorizer, nova_frase)
