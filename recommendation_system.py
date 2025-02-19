import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Carregar os dados
clientes_df = pd.read_csv("data/clientes.csv")
tenis_df = pd.read_csv("data/tenis.csv")
compras_df = pd.read_csv("data/compras.csv")

def recomendar_tenis(cliente_id):
    cliente = clientes_df[clientes_df['ID_Cliente'] == cliente_id].iloc[0]
    idade = cliente['Idade']
    genero = cliente['Genero']
    publico = "Adulto" if idade >= 18 else "Kids"

    # historico de compras do cliente
    compras_cliente = compras_df[compras_df['ID_Cliente'] == cliente_id]
    tenis_comprados = compras_cliente['ID_Tenis'].tolist()

    if compras_cliente.empty:
        # caso o cliente nao tenha historico, recomendar um tenis aleatorio
        tenis_disponiveis = tenis_df[(tenis_df['Publico'] == publico) & (tenis_df['Genero'].isin([genero, "Unissex"]))]
        if tenis_disponiveis.empty:
            return "Nenhuma recomendação encontrada para este cliente."
        return tenis_disponiveis.sample(1)[['ID_Tenis', 'Nome', 'Categoria', 'Genero', 'Publico', 'Cor']]

    # filtrar tenis disponiveis
    tenis_disponiveis = tenis_df[(tenis_df['Publico'] == publico) & (tenis_df['Genero'].isin([genero, "Unissex"]))]
    tenis_disponiveis = tenis_disponiveis[~tenis_disponiveis['ID_Tenis'].isin(tenis_comprados)].reset_index(drop=True) # reset index aqui

    if tenis_disponiveis.empty:
        return "Nenhuma recomendação encontrada para este cliente."

    # criar vetorizadores
    vectorizer_geral = CountVectorizer(token_pattern=r"(?u)\b\w+\b")
    vectorizer_cor = CountVectorizer(token_pattern=r"(?u)\b\w+\b")
    vectorizer_categoria = CountVectorizer(token_pattern=r"(?u)\b\w+\b")

    # obter dados dos tenis comprados
    tenis_comprados_df = tenis_df[tenis_df['ID_Tenis'].isin(tenis_comprados)]

    # criar matriz de caracteristicas para os tenis disponiveis
    caracteristicas_geral = tenis_disponiveis[['Categoria', 'Cor']].apply(lambda x: ' '.join(x), axis=1)
    caracteristicas_cor = tenis_disponiveis['Cor'] # apenas a cor
    caracteristicas_categoria = tenis_disponiveis['Categoria'] # apenas a categoria

    caracteristicas_matrix_geral = vectorizer_geral.fit_transform(caracteristicas_geral)
    caracteristicas_matrix_cor = vectorizer_cor.fit_transform(caracteristicas_cor)
    caracteristicas_matrix_categoria = vectorizer_categoria.fit_transform(caracteristicas_categoria)

    # criar matriz de caracteristicas para os tenis comprados
    caracteristicas_compradas_geral = tenis_comprados_df[['Categoria', 'Cor']].apply(lambda x: ' '.join(x), axis=1)
    caracteristicas_compradas_cor = tenis_comprados_df['Cor'] # apenas a cor
    caracteristicas_compradas_categoria = tenis_comprados_df['Categoria'] # apenas a categoria

    caracteristicas_compradas_matrix_geral = vectorizer_geral.transform(caracteristicas_compradas_geral)
    caracteristicas_compradas_matrix_cor = vectorizer_cor.transform(caracteristicas_compradas_cor)
    caracteristicas_compradas_matrix_categoria = vectorizer_categoria.transform(caracteristicas_compradas_categoria)

    # calcular similaridades
    if not tenis_comprados_df.empty: # verifica se ha compras para calcular similaridade
        similarity_geral = cosine_similarity(caracteristicas_compradas_matrix_geral, caracteristicas_matrix_geral).mean(axis=0)
        similarity_cor = cosine_similarity(caracteristicas_compradas_matrix_cor, caracteristicas_matrix_cor).mean(axis=0)
        similarity_categoria = cosine_similarity(caracteristicas_compradas_matrix_categoria, caracteristicas_matrix_categoria).mean(axis=0)
    else:
        similarity_geral = np.zeros(caracteristicas_matrix_geral.shape[0]) # caso nao haja compras, similaridade zero
        similarity_cor = np.zeros(caracteristicas_matrix_cor.shape[0])
        similarity_categoria = np.zeros(caracteristicas_matrix_categoria.shape[0])


    # criar tabelas ordenadas por similaridade
    def gerar_recomendacoes(similarity_scores):
        sorted_indices = np.argsort(similarity_scores)[::-1]  # Oordena por similaridade decrescente
        recomendacoes_ids = []
        recomendacoes_data = []

        for index in sorted_indices:
            tenis_id = tenis_disponiveis.iloc[index]['ID_Tenis']
            tamanho_comprado = compras_cliente['Tamanho_Comprado'].tolist()

            tem_no_estoque = any(str(tam) in tenis_disponiveis.iloc[index]['Estoque'] for tam in tamanho_comprado)

            if tem_no_estoque and tenis_id not in recomendacoes_ids: # adicionado para evitar duplicacoes e garantir estoque
                recomendacoes_ids.append(tenis_id)
                recomendacoes_data.append(tenis_disponiveis.iloc[index])

        return pd.DataFrame(recomendacoes_data)[['ID_Tenis', 'Nome', 'Categoria', 'Genero', 'Publico', 'Cor']] if recomendacoes_data else pd.DataFrame()


    # tabelas de recomendacao
    recomendacoes_geral = gerar_recomendacoes(similarity_geral)
    recomendacoes_cor = gerar_recomendacoes(similarity_cor) # usando similarity_cor diretamente
    recomendacoes_categoria = gerar_recomendacoes(similarity_categoria) # usando similarity_categoria diretamente

    return recomendacoes_geral, recomendacoes_cor, recomendacoes_categoria


geral, cor, categoria = recomendar_tenis(1000)

if not geral.empty:
    print("\nGeral:")
    print(geral.head(3))
else:
    print("\nGeral:")
    print("Nenhuma recomendacao geral.")

if not cor.empty:
    print("\nCor:")
    print(cor.head(3))
else:
    print("\nCor:")
    print("Nenhuma recomendacao por cor.")

if not categoria.empty:
    print("\nCategoria:")
    print(categoria.head(3))
else:
    print("\nCategoria:")
    print("Nenhuma recomendacao por categoria.")