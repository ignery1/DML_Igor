import pandas as pd
import streamlit as st
import joblib

# Carregando o modelo treinado
model = joblib.load('model.pkl')

# Carregando os dados dos estudantes
dataset = pd.read_csv('estudantes.csv')

# Criando um dicionário para mapear os valores das variáveis categóricas
categorical_mapping = {
    'genero': {'Feminino': 1, 'Masculino': 0},
    'etinia': {'A': 1, 'B': 0, 'C': 0, 'D': 0, 'E': 0},
    'educacao_pais': {'AD': 1, 'BD': 0, 'HS': 0, 'SC': 0, 'SHS': 0},
    'curso_preparacao': {'Completo': 1, 'Nenhum': 0},
    'almoco': {'Gratuito/Reduzido': 1, 'Padrão': 0}
}

# Criando a interface do Streamlit
st.title("Predição de notas de matemática")
st.markdown("Este é um aplicativo que realiza a predição da nota de matemática com base nos atributos do aluno.")

# Criando os campos de entrada para os atributos do aluno
atributos_aluno = {
    'nota_leitura': st.sidebar.number_input("Nota de Leitura"),
    'nota_escrita': st.sidebar.number_input("Nota de Escrita"),
    'genero': st.sidebar.selectbox("Gênero do Aluno", ('Feminino', 'Masculino')),
    'etinia': st.sidebar.selectbox("Raça/Etnia", ('A', 'B', 'C', 'D', 'E')),
    'educacao_pais': st.sidebar.selectbox("Grau de Escolaridade", ('BD', 'SC', 'MD', 'AD', 'HS', 'SHS')),
    'curso_preparacao': st.sidebar.selectbox("Curso Preparatório para Teste", ('Nenhum', 'Completo')),
    'almoco': st.sidebar.selectbox("Tipo de Almoço", ('Gratuito/Reduzido', 'Padrão'))
}

# Realizando a predição quando o botão for acionado
btn_predict = st.sidebar.button("Realizar Predição")

if btn_predict:
    data_teste = pd.DataFrame([atributos_aluno])

    # Mapeando os valores categóricos para numéricos
    for col, mapping in categorical_mapping.items():
        data_teste[col] = data_teste[col].map(mapping)

    result = model.predict(data_teste)
    result = round(result[0], 2)

    st.subheader("Nota de matemática predita:")
    st.write(result)
