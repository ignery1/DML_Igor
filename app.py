import pandas as pd
import streamlit as st
from pycaret.regression import load_model

# Carregar o modelo treinado
model = load_model('model')

# Carregar o conjunto de dados
dataset = pd.read_csv('estudantes.csv') 

# Configurar a interface do Streamlit
st.title("Predição de notas de matemática")
st.markdown("Este é um aplicativo de dados usado para prever notas de matemática usando Machine Learning.")

# Definir os atributos do aluno para a predição da nota de matemática
nota_leitura = st.sidebar.number_input("Nota de Leitura")
nota_escrita = st.sidebar.number_input("Nota de Escrita")
genero = st.sidebar.selectbox("Gênero do Aluno", ("Feminino", "Masculino"))
etinia = st.sidebar.selectbox("Raça/Etnia", ("A", "B", "C", "D", "E"))
educacao_pais = st.sidebar.selectbox("Grau de Escolaridade dos Pais", ("AD", "BD", "HS", "SC", "SHS"))
curso_preparacao = st.sidebar.selectbox("Curso Preparatório para Teste", ("Nenhum", "Completo"))
almoco = st.sidebar.selectbox("Tipo de Almoço", ("Gratuito/Reduzido", "Padrão"))

# Mapear os valores selecionados para os atributos correspondentes
gender_mapping = {"Feminino": 1, "Masculino": 0}
ethnicity_mapping = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5}
education_mapping = {"AD": 1, "BD": 2, "HS": 3, "SC": 4, "SHS": 5}
prep_course_mapping = {"Nenhum": 0, "Completo": 1}
lunch_mapping = {"Gratuito/Reduzido": 0, "Padrão": 1}

# Criar o dataframe de teste com os valores selecionados
data_teste = pd.DataFrame({
    "nota_leitura": [nota_leitura],
    "nota_escrita": [nota_escrita],
    "female": [gender_mapping[genero]],
    "male": [1 - gender_mapping[genero]],
    "A": [1 if etinia == "A" else 0],
    "B": [1 if etinia == "B" else 0],
    "C": [1 if etinia == "C" else 0],
    "D": [1 if etinia == "D" else 0],
    "E": [1 if etinia == "E" else 0],
    "AD": [1 if educacao_pais == "AD" else 0],
    "BD": [1 if educacao_pais == "BD" else 0],
    "HS": [1 if educacao_pais == "HS" else 0],
    "SC": [1 if educacao_pais == "SC" else 0],
    "SHS": [1 if educacao_pais == "SHS" else 0],
    "FR": [1 if almoco == "Gratuito/Reduzido" else 0],
    "S": [1 if almoco == "Padrão" else 0],
    "completed": [prep_course_mapping[curso_preparacao]],
    "none": [1 - prep_course_mapping[curso_preparacao]]
})

# Realizar a predição
result = model.predict(data_teste)
predicted_math_score = round(result[0], 2)

# Exibir a nota de matemática predita
st.subheader("Nota de Matemática Predita:")
st.write(predicted_math_score)
