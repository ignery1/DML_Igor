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
educacao_pais = st.sidebar.selectbox("Grau de Escolaridade dos Pais", ("BD", "SC", "MD", "AD", "HS", "SHS"))
curso_preparacao = st.sidebar.selectbox("Curso Preparatório para Teste", ("Nenhum", "Completo"))
almoco = st.sidebar.selectbox("Tipo de Almoço", ("Gratuito/Reduzido", "Padrão"))

# Mapear os valores selecionados para os atributos correspondentes
gender_mapping = {"Feminino": 1, "Masculino": 0}
ethnicity_mapping = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5}
education_mapping = {"BD": 1, "SC": 2, "MD": 3, "AD": 4, "HS": 5, "SHS": 6}
prep_course_mapping = {"Nenhum": 0, "Completo": 1}
lunch_mapping = {"Gratuito/Reduzido": 0, "Padrão": 1}

# Criar o dataframe de teste com os valores selecionados
data_teste = pd.DataFrame({
    "nota_leitura": [nota_leitura],
    "nota_escrita": [nota_escrita],
    "gender": [gender_mapping[genero]],
    "ethnicity": [ethnicity_mapping[etinia]],
    "parental_education": [education_mapping[educacao_pais]],
    "prep_course": [prep_course_mapping[curso_preparacao]],
    "lunch": [lunch_mapping[almoco]]
})

# Realizar a predição usando o modelo carregado
result = model.predict(data_teste)
predicted_score = round(result[0], 2)

# Exibir o resultado da predição
st.subheader("Nota de Matemática Predita:")
st.write(predicted_score)
