#!/usr/bin/env python
# coding: utf-8

# # Desafio 6
# 
# Neste desafio, vamos praticar _feature engineering_, um dos processos mais importantes e trabalhosos de ML. Utilizaremos o _data set_ [Countries of the world](https://www.kaggle.com/fernandol/countries-of-the-world), que contém dados sobre os 227 países do mundo com informações sobre tamanho da população, área, imigração e setores de produção.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Setup_ geral

# In[17]:


import pandas as pd
import numpy as np
import seaborn as sns
import sklearn as sk
from sklearn.preprocessing import KBinsDiscretizer as KB
from sklearn.preprocessing import OneHotEncoder as OHE
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


# In[2]:


# Algumas configurações para o matplotlib.

from IPython.core.pylabtools import figsize


figsize(12, 8)

sns.set()


# In[77]:


countries = pd.read_csv("countries.csv")


# In[78]:


new_column_names = [
    "Country", "Region", "Population", "Area", "Pop_density", "Coastline_ratio",
    "Net_migration", "Infant_mortality", "GDP", "Literacy", "Phones_per_1000",
    "Arable", "Crops", "Other", "Climate", "Birthrate", "Deathrate", "Agriculture",
    "Industry", "Service"
]

countries.columns = new_column_names
df = countries.copy()


# ## Observações
# 
# Esse _data set_ ainda precisa de alguns ajustes iniciais. Primeiro, note que as variáveis numéricas estão usando vírgula como separador decimal e estão codificadas como strings. Corrija isso antes de continuar: transforme essas variáveis em numéricas adequadamente.
# 
# Além disso, as variáveis `Country` e `Region` possuem espaços a mais no começo e no final da string. Você pode utilizar o método `str.strip()` para remover esses espaços.

# ## Inicia sua análise a partir daqui

# ## Questão 1
# 
# Quais são as regiões (variável `Region`) presentes no _data set_? Retorne uma lista com as regiões únicas do _data set_ com os espaços à frente e atrás da string removidos (mas mantenha pontuação: ponto, hífen etc) e ordenadas em ordem alfabética.

# In[448]:


def q1():
    regioes = list(df['Region'].sort_values().unique())
    return(list(map(lambda regioes: regioes.strip(), regioes)))


# ## Questão 2
# 
# Discretizando a variável `Pop_density` em 10 intervalos com `KBinsDiscretizer`, seguindo o encode `ordinal` e estratégia `quantile`, quantos países se encontram acima do 90º percentil? Responda como um único escalar inteiro.

# In[436]:


def q2():
    # Trocando , por .
    df['Pop_density'] = df['Pop_density'].replace(',','.',regex=True)
    # Criando uma cópia do dataset
    pop = df[['Pop_density']].copy()
    # Criando KBinsDiscretizer
    est = KB(n_bins=10, encode='ordinal', strategy='quantile')
    x = est.fit_transform(pop)
    # Adicionando a nova feature no dataset
    df['Density_BIN'] = x
    # Calculando a quantidade de valores acima do 90° percentil
    q_90 = df['Density_BIN'].quantile(.90)
    return(int((df['Density_BIN'] > q_90).sum()))


# # Questão 3
# 
# Se codificarmos as variáveis `Region` e `Climate` usando _one-hot encoding_, quantos novos atributos seriam criados? Responda como um único escalar.

# In[9]:


def q3():
    ## Tratando Climate ##
    climate = df['Climate'].copy()
    # Trocar ',' por '.'
    climate = climate.replace(',','.',regex=True)
    # Transformar de object para float
    climate = climate.astype(float)
    # Substituindo Nan por -1
    climate = climate.fillna(-1)
    return(climate.nunique() + df['Region'].nunique())


# ## Questão 4
# 
# Aplique o seguinte _pipeline_:
# 
# 1. Preencha as variáveis do tipo `int64` e `float64` com suas respectivas medianas.
# 2. Padronize essas variáveis.
# 
# Após aplicado o _pipeline_ descrito acima aos dados (somente nas variáveis dos tipos especificados), aplique o mesmo _pipeline_ (ou `ColumnTransformer`) ao dado abaixo. Qual o valor da variável `Arable` após o _pipeline_? Responda como um único float arredondado para três casas decimais.

# In[79]:


test_country = [
    'Test Country', 'NEAR EAST', -0.19032480757326514,
    -0.3232636124824411, -0.04421734470810142, -0.27528113360605316,
    0.13255850810281325, -0.8054845935643491, 1.0119784924248225,
    0.6189182532646624, 1.0074863283776458, 0.20239896852403538,
    -0.043678728558593366, -0.13929748680369286, 1.3163604645710438,
    -0.3699637766938669, -0.6149300604558857, -0.854369594993175,
    0.263445277972641, 0.5712416961268142
]


# In[176]:


def q4():
    # Transformando a coluna Arable em float64
    new_countries = countries.replace(',','.',regex=True)
    new_countries['Arable'] = new_countries['Arable'].astype('float64')
    # Coletando as colunas com int64 e float64
    columns = list(new_countries.select_dtypes(include = "number").columns)
    # Criando um dataset com test_country
    country_t = (pd.DataFrame(test_country,index=countries.columns)).transpose()
    # Criando o pipeline
    pipeline = Pipeline(steps=[('median_imputer', SimpleImputer(strategy='median')),('std_sca',StandardScaler())])
    # Aplicando a Pipeline no test_country
    pipeline.fit(new_countries[columns])
    # Retornando o valor de Arable
    return(float(round(pipeline.transform(country_t[columns])[0][3],3)))


# ## Questão 5
# 
# Descubra o número de _outliers_ da variável `Net_migration` segundo o método do _boxplot_, ou seja, usando a lógica:
# 
# $$x \notin [Q1 - 1.5 \times \text{IQR}, Q3 + 1.5 \times \text{IQR}] \Rightarrow x \text{ é outlier}$$
# 
# que se encontram no grupo inferior e no grupo superior.
# 
# Você deveria remover da análise as observações consideradas _outliers_ segundo esse método? Responda como uma tupla de três elementos `(outliers_abaixo, outliers_acima, removeria?)` ((int, int, bool)).

# In[460]:


def q5():
    # Trocando ',' por '.' e modificando para float.
    df['Net_migration'] = df['Net_migration'].replace(',','.',regex=True).astype(float)
    # Calculando Q1, Q3 e IQR.
    Q3 = df['Net_migration'].quantile(.75)
    Q1 = df['Net_migration'].quantile(.25)
    IQR = Q3 - Q1
    # Calculando Outliers
    outl_max = df[df['Net_migration'] > Q3 + 1.5*IQR]['Net_migration']
    outl_min = df[df['Net_migration'] < Q1 - 1.5*IQR]['Net_migration'] 
    # Resultado
    return(outl_min.shape[0],outl_max.shape[0],False)
    # Não devemos retirar os "outliers", pois, nesse caso, eles são o "normal" do dataset.


# ## Questão 6
# Para as questões 6 e 7 utilize a biblioteca `fetch_20newsgroups` de datasets de test do `sklearn`
# 
# Considere carregar as seguintes categorias e o dataset `newsgroups`:
# 
# ```
# categories = ['sci.electronics', 'comp.graphics', 'rec.motorcycles']
# newsgroup = fetch_20newsgroups(subset="train", categories=categories, shuffle=True, random_state=42)
# ```
# 
# 
# Aplique `CountVectorizer` ao _data set_ `newsgroups` e descubra o número de vezes que a palavra _phone_ aparece no corpus. Responda como um único escalar.

# In[180]:


def q6():
    categories = ['sci.electronics', 'comp.graphics', 'rec.motorcycles']
    newsgroup = fetch_20newsgroups(subset="train", categories=categories, shuffle=True, random_state=42)
    # "Vectorizando" o data set
    counter = CountVectorizer()
    freq = counter.fit_transform(newsgroup.data)
    # Recebendo o vocabulário
    words = dict(counter.vocabulary_.items())
    # Retornando a soma.
    return(int(freq[:,words['phone']].sum()))


# ## Questão 7
# 
# Aplique `TfidfVectorizer` ao _data set_ `newsgroups` e descubra o TF-IDF da palavra _phone_. Responda como um único escalar arredondado para três casas decimais.

# In[183]:


def q7():
    # Criando o data set
    categories = ['sci.electronics', 'comp.graphics', 'rec.motorcycles']
    newsgroup = fetch_20newsgroups(subset="train", categories=categories, shuffle=True, random_state=42)
    # "Vectorizando" o data set
    Tfid = TfidfVectorizer()
    freq = Tfid.fit_transform(newsgroup.data)
    # Retornando o idf
    return(round(float(freq[:,Tfid.vocabulary_['phone']].sum()),3))


# In[184]:


q7()


# In[ ]:




