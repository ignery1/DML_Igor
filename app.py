import pandas as pd
import streamlit as st
import joblib

# Carregando o modelo treinado
model = joblib.load('model.pkl')

# Carregando os dados dos estudantes
dataset = pd.read_csv('estudantes.csv')
print(dataset.columns)


# Criando um dicionário para mapear os valores das variáveis categóricas
categorical_mapping = {
    'female': {'Feminino': 1, 'Masculino': 0},
    'A': {'A': 1, 'B': 0, 'C': 0, 'D': 0, 'E': 0},
    'AD': {'AD': 1, 'BD': 0, 'HS': 0, 'SC': 0, 'SHS': 0},
    'completed': {'Completo': 1, 'Nenhum': 0},
    'FR': {'Gratuito/Reduzido': 1, 'Padrão': 0}
}

# Criando a interface do Streamlit
st.title("Predição de notas de matemática")
st.markdown("Este é um aplicativo que realiza a predição da nota de matemática com base nos atributos do aluno.")

# Criando os campos de entrada para os atributos do aluno
atributos_aluno = {
    'nota_leitura': st.sidebar.number_input("Nota de Leitura"),
    'nota_escrita': st.sidebar.number_input("Nota de Escrita"),
    'female': st.sidebar.selectbox("Gênero do Aluno", ('Feminino', 'Masculino')),
    'A': st.sidebar.selectbox("Raça/Etnia", ('A', 'B', 'C', 'D', 'E')),
    'AD': st.sidebar.selectbox("Grau de Escolaridade", ('BD', 'SC', 'MD', 'AD', 'HS', 'SHS')),
    'completed': st.sidebar.selectbox("Curso Preparatório para Teste", ('Nenhum', 'Completo')),
    'FR': st.sidebar.selectbox("Tipo de Almoço", ('Gratuito/Reduzido', 'Padrão'))
}

# Selecionando as colunas relevantes do dataset
colunas_relevantes = ['nota_leitura', 'nota_escrita', 'female', 'A', 'AD', 'completed', 'FR']
#data_teste = dataset[colunas_relevantes].copy()
data_teste = dataset.loc[:, colunas_relevantes].copy()

# Preenchendo o data_teste com base nos atributos do aluno
data_teste = data_teste.append(atributos_aluno, ignore_index=True)

# Mapeando os valores categóricos para numéricos
for col, mapping in categorical_mapping.items():
    data_teste[col] = data_teste[col].map(mapping)

# Realizando a predição quando o botão for acionado
btn_predict = st.sidebar.button("Realizar Predição")

if btn_predict:
    result = model.predict(data_teste.tail(1))
    result = round(result[0], 2)

    st.subheader("Nota de matemática predita:")
    st.write(result)
