import pandas as pd
import streamlit as st
from pycaret.regression import *
import numpy as np
 

# loading the trained model.
model = load_model('modelo_pronto')

# carregando uma amostra dos dados.
dataset = pd.read_csv('estudantes.csv') 
#classifier = pickle.load(pickle_in)

# título
st.title("Predição de notas de matemática")

# subtítulo
st.markdown("Este é um Data App utilizado para exibir a solução de Machine Learning para o problema de predição de notas de matemática.")



st.sidebar.subheader("Defina os atributos do aluno para a predição da nota de matemática")


# mapeando dados do usuário para cada atributo
nota_leitura = st.sidebar.number_input("Nota de Leitura")
nota_escrita = st.sidebar.number_input("Nota de Escrita")


genero = st.sidebar.selectbox("Gênero do Aluno",("Feminino","Masculino"))
etinia = st.sidebar.selectbox("Raça/Etinia",("A","B","C","D","E"))
educacao_pais = st.sidebar.selectbox("Grau de Escolaridade",("BD","SC","MD","AD","HS", "SHS"))
curso_preparacao = st.sidebar.selectbox("Curso Preparatório para Teste",("Nenhum","Completo"))
almoco = st.sidebar.selectbox("Tipo de Almoço",("Gratuito/Reduzido","Padrão"))

# transformando o dado de entrada em valor binário
female = 1 if genero == "Feminino" else 0
male = 1 if genero == "Masculino" else 0

A = 1 if etinia == "A" else 0
B = 1 if etinia == "B" else 0
C = 1 if etinia == "C" else 0
D = 1 if etinia == "D" else 0
E = 1 if etinia == "E" else 0

AD = 1 if educacao_pais == "AD" else 0
BD = 1 if educacao_pais == "BD" else 0
HS = 1 if educacao_pais == "HS" else 0
SC = 1 if educacao_pais == "SC" else 0
SHS = 1 if educacao_pais == "SHS" else 0


completed = 1 if curso_preparacao == "Completo" else 0
none = 1 if curso_preparacao == "Nenhum" else 0

FR = 1 if almoco == "Gratuito/Reduzido" else 0
S = 1 if almoco == "Padrão" else 0

# inserindo um botão na tela
btn_predict = st.sidebar.button("Realizar Predição")

# verifica se o botão foi acionado
if btn_predict:
    data_teste = pd.DataFrame()

    data_teste["nota_leitura"] =	[nota_leitura]
    data_teste["nota_escrita"] =	[nota_escrita]    
    data_teste["female"] = [female]
    data_teste["male"] = [male]	
    data_teste["A"] = [A]
    data_teste["B"] = [B]
    data_teste["C"] = [C]
    data_teste["D"] =	[D]
    data_teste["E"] =	[E]
    data_teste["AD"] = [AD]
    data_teste["BD"] = [BD]
    data_teste["HS"] = [HS]
    data_teste["SC"] = [SC]
    data_teste["SHS"] = [SHS]
    data_teste["FR"] = [FR]
    data_teste["S"] = [S]
    data_teste["completed"] = [completed]
    data_teste["none"] = [none]


    #imprime os dados de teste    
    print(data_teste)

    #realiza a predição
    result = model.predict(data_teste)
    
    st.subheader("Nota de matematica predita:")
    result = (round(result[0],2))
    
    st.write(result)
   
