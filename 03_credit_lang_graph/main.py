# ==================================================
# CONFIGURAÇÃO E CAMINHOS
# ==================================================
import os
from pathlib import Path
from dotenv import load_dotenv

# ==================================================
# VALORES FINANCEIROS
# ==================================================
from decimal import Decimal, ROUND_HALF_UP

# ==================================================
# VALIDAÇÃO COM PYDANTIC
# ==================================================
from pydantic import BaseModel, Field, ValidationError

# ==================================================
# MODELO, PROMPTS E FERRAMENTAS
# ==================================================
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_classic.agents import (
    AgentExecutor,
    create_tool_calling_agent,
)
from langchain_core.prompts import MessagesPlaceholder

# ==================================================
# DOCUMENTOS E DIVISÃO EM CHUNKS
# ==================================================
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ==================================================
# EMBEDDINGS E BANCO VETORIAL
# ==================================================
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ==================================================
# ESTADO E GRAFO
# ==================================================
import operator
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END

# ==================================================
# CAMINHOS DO PROJETO
# ==================================================
CAMINHO_PASTA = Path(__file__).resolve().parent / "documentos"

# ==================================================
# 1. CARREGAR DOCUMENTOS
# ==================================================

def carregar_documentos():
    """Lê toda a pasta de documentos, pegando cada um dos PDFs."""

    arquivos = sorted(CAMINHO_PASTA.glob("*.pdf"))

    if not arquivos:
        raise ValueError("Nenhum arquivo PDF encontrado.")

    documentos = []

    for arquivo in arquivos:
        carregar = PyPDFLoader(str(arquivo))
        paginas = carregar.load()

        for pagina in paginas:
            if not pagina.page_content.strip():
                continue

            # Pegando o nome da fonte
            pagina.metadata["fonte"] = arquivo.name
            # Pegando o número da página
            pagina.metadata["pagina"] = pagina.metadata["page"] + 1
            # Pegando o título
            pagina.metadata["titulo"] = (pagina.metadata.get("title") or arquivo.stem)
            # Pegando o assunto
            pagina.metadata["assunto"] = (pagina.metadata.get("subject") or "Não informado")

            documentos.append(pagina)

    print(f"Total de PDFs lido -> {len(arquivos)}")
    print(f"Total de Páginas encontradas -> {len(documentos)}")

    return documentos

# ==================================================
# 2. DIVIDIR DOCUMENTOS EM CHUNKS
# ==================================================

def dividir_documentos(documentos):
    """Dividindo os documentos em chunks."""

    divisor = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=350
    )

    trechos = divisor.split_documents(documentos)

    print(f"Total de Trechos criados -> {len(trechos)}")

    return trechos

# ==================================================
# 3. CRIANDO BANCO + EMBEDDINGS
# ==================================================

def criar_banco(trechos):
    """Criando embeddings e o banco vetorial."""

    embeddings = HuggingFaceEmbeddings(
        model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    banco = FAISS.from_documents(
        documents=trechos,
        embedding=embeddings
    )

    return banco

# ==================================================
# 4. RECUPERANDO O CONTEXTO
# ==================================================

def recuperar_contexto(pergunta, banco):
    """Recuperando o contexto por busca semântica."""

    documentos = banco.similarity_search(
        pergunta,
        k=5
    )

    partes = []

    for numero, documento in enumerate(documentos, start=1):

        parte =(
            f"\nTrecho {numero}\n"
            f"Fonte: {documento.metadata["fonte"]}\n"
            f"Página: {documento.metadata["pagina"]}\n"
            f"Título: {documento.metadata["titulo"]}\n"
            f"Assunto: {documento.metadata["assunto"]}\n"
            f"Conteúdo: {documento.page_content}"
        )

        partes.append(parte)

    contexto = "\n\n".join(partes)

    return contexto

# ==================================================
# 5. CRIANDO O MODELO
# ==================================================

def criar_modelo():
    """Criando o modelo GROQ para chamar nas respostas."""

    load_dotenv()

    if not os.getenv("GROQ_API_KEY"):
        raise ValueError("Configure GROQ_API_KEY no arquivo .env")

    modelo = ChatGroq(model="openai/gpt-oss-20b", temperature=0)

    return modelo

# ==================================================
# 6. CRIANDO CLASSE PYDANTIC
# ==================================================

class RespostaRag(BaseModel):
    """Organizando a resposta com base nos documentos recuperados."""

    assunto: str = Field(
        description=(
            "Assunto principal identificado no contexto. Retorne vazio se não encontrar."
        )
    )

    resposta: str = Field(
        description=(
            "Resposta objetiva à pergunta, baseada exclusivamente no contexto recuperado. "
            "Caso o contexto não ter base suficiente, apenas retorne que não foi possível "
            "responder a pergunta que o cliente fez."
        )
    )

    fontes : str = Field(
        description=(
            "Retorne as Fontes que o contexto conseguiu recuperar. "
            "Caso não tenha, retorne null"
        )
    )

# ==================================================
# 7. CRIANDO CHAIN DO RAG
# ==================================================

def criar_chain_rag(modelo):
    """Responde a pergunta com base no contexto recuperado."""

documentos = carregar_documentos()
trechos = dividir_documentos(documentos)
banco = criar_banco(trechos)

pergunta = "gostaria de solicitar um credito"
contexto = recuperar_contexto(pergunta, banco)
print(contexto)