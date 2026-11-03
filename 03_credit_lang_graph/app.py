# ==================================================
# 1. IMPORTAÇÕES
# ==================================================

import streamlit as st

from langchain_core.messages import HumanMessage, AIMessage


# ==================================================
# 2. CONFIGURAÇÃO DA PÁGINA
# ==================================================

st.set_page_config(
    page_title="Aurora | Assistente do Cooperado",
    page_icon="🏦",
    layout="centered"
)


# ==================================================
# 3. APRESENTAÇÃO DO PROJETO
# ==================================================

st.markdown(
    """
    <style>
        .aurora-cabecalho {
            padding: 30px 28px;
            margin-bottom: 22px;
            border: 1px solid #285a55;
            border-radius: 18px;
            background: linear-gradient(135deg, #123b36, #101f2b);
        }

        .aurora-etiqueta {
            color: #99f6e4;
            font-size: 12px;
            font-weight: 700;
            letter-spacing: 2px;
            margin-bottom: 12px;
        }

        .aurora-titulo {
            color: #f8fafc;
            font-size: clamp(28px, 5vw, 40px);
            font-weight: 750;
            line-height: 1.2;
            margin: 0 0 14px;
        }

        .aurora-descricao {
            color: #cbd5e1;
            font-size: 16px;
            line-height: 1.6;
            margin: 0;
        }

        .aurora-temas {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-top: 22px;
        }

        .aurora-tema {
            color: #ccfbf1;
            background: #19443f;
            border: 1px solid #30635b;
            border-radius: 20px;
            padding: 5px 12px;
            font-size: 12px;
        }
    </style>

    <section class="aurora-cabecalho">
        <div class="aurora-etiqueta">COOPERATIVA AURORA</div>
        <h1 class="aurora-titulo">
            Tire dúvidas. Entenda sua proposta.
        </h1>
        <p class="aurora-descricao">
            Consulte informações da cooperativa e calcule
            indicadores de crédito. Continue a conversa para
            esclarecer dúvidas ou atualizar os dados da sua proposta.
        </p>
        <div class="aurora-temas">
            <span class="aurora-tema">Associação e cadastro</span>
            <span class="aurora-tema">Produtos e serviços</span>
            <span class="aurora-tema">Indicadores financeiros</span>
        </div>
    </section>
    """,
    unsafe_allow_html=True
)


# ==================================================
# 4. HISTÓRICO DA SESSÃO
# ==================================================

if "mensagens" not in st.session_state:
    st.session_state.mensagens = []

with st.sidebar:
    st.header("Conversa")

    if st.button("Nova conversa", use_container_width=True):
        st.session_state.mensagens = []
        st.rerun()

    st.caption(
        "O histórico é utilizado para compreender perguntas "
        "complementares e alterações na mesma proposta."
    )

    st.divider()

    st.caption(
        "Projeto educacional com documentos de uma cooperativa "
        "fictícia. Não representa políticas de instituições reais."
    )

    st.caption(
        "O assistente não aprova crédito, não acessa contas "
        "e não executa operações financeiras."
    )


# ==================================================
# 5. SUGESTÕES INICIAIS
# ==================================================

if not st.session_state.mensagens:
    st.markdown("**Por onde começar?**")

    st.markdown(
        """
        - Quais documentos preciso apresentar para me associar como MEI?
        - O que fazer se eu estiver com dificuldades para pagar?
        - Minha renda líquida é R$ 5.500 e pago R$ 700 em parcelas.
          Qual é meu comprometimento atual?
        """
    )

    st.caption(
        "Durante a conversa, você pode atualizar os dados: "
        "“Corrigindo: minha renda líquida é R$ 6.000”."
    )


# ==================================================
# 6. CARREGAR O GRAFO
# ==================================================

@st.cache_resource
def carregar_chatbot():
    """Prepara e reutiliza o grafo entre as interações."""

    # Importação aqui para incluir a preparação dos componentes
    # dentro do carregamento inicial da aplicação.
    from main import criar_grafo

    return criar_grafo()


try:
    with st.spinner("Preparando o assistente e os documentos..."):
        chatbot = carregar_chatbot()

except Exception as erro:
    st.error(
        "Não foi possível iniciar o assistente. "
        "Confira a configuração da API e a pasta de documentos."
    )

    with st.expander("Detalhes do erro"):
        st.exception(erro)

    st.stop()


# ==================================================
# 7. EXIBIR UMA MENSAGEM
# ==================================================

def exibir_mensagem(mensagem):
    """Mostra a mensagem e os detalhes disponíveis."""

    with st.chat_message(mensagem["papel"]):
        if mensagem.get("assunto"):
            st.caption(mensagem["assunto"])

        st.markdown(mensagem["conteudo"])

        if mensagem.get("contexto"):
            with st.expander("Trechos recuperados dos documentos"):
                st.text(mensagem["contexto"])

        if mensagem.get("proposta") is not None:
            with st.expander("Dados utilizados nos cálculos"):
                st.json(mensagem["proposta"])

        if mensagem.get("trilha"):
            with st.expander("Caminho percorrido"):
                for etapa in mensagem["trilha"]:
                    st.write(f"- {etapa}")


for mensagem in st.session_state.mensagens:
    exibir_mensagem(mensagem)


# ==================================================
# 8. RECEBER A SOLICITAÇÃO
# ==================================================

mensagem = st.chat_input(
    "Qual dúvida ou cálculo você precisa?"
)

if mensagem and mensagem.strip():
    mensagem = mensagem.strip()

    mensagem_usuario = {
        "papel": "user",
        "conteudo": mensagem
    }

    exibir_mensagem(mensagem_usuario)

    # Converter a conversa anterior para o formato do LangChain.
    # A mensagem atual será enviada separadamente no estado.
    historico = []

    for mensagem_anterior in st.session_state.mensagens:
        if mensagem_anterior["papel"] == "user":
            historico.append(
                HumanMessage(
                    content=mensagem_anterior["conteudo"]
                )
            )
        else:
            historico.append(
                AIMessage(
                    content=mensagem_anterior["conteudo"]
                )
            )

    entrada = {
        "mensagem": mensagem,
        "historico": historico,
        "mensagem_interpretada": "",
        "classificacao": None,
        "contexto": "",
        "proposta": None,
        "resposta_rag": None,
        "resultados_calculos": "",
        "resposta_final": "",
        "trilha": []
    }


    # ==================================================
    # 9. EXECUTAR O GRAFO E APRESENTAR A RESPOSTA
    # ==================================================

    try:
        with st.spinner("Analisando sua solicitação..."):
            resultado = chatbot.invoke(entrada)

        resposta_final = resultado["resposta_final"]

        if not resposta_final or not resposta_final.strip():
            st.error(
                "Não foi possível gerar uma resposta. Tente novamente."
            )

        else:
            # Adaptar os títulos do terminal para a interface.
            resposta_formatada = resposta_final.replace(
                "=== CONSULTA AOS DOCUMENTOS ===",
                "### Consulta aos documentos"
            ).replace(
                "=== INDICADORES FINANCEIROS ===",
                "### Indicadores financeiros"
            )

            consulta = resultado["resposta_rag"]
            proposta = resultado["proposta"]

            mensagem_assistente = {
                "papel": "assistant",
                "assunto": consulta.assunto if consulta is not None else "",
                "conteudo": resposta_formatada,
                "contexto": resultado["contexto"],
                "proposta": (
                    proposta.model_dump(mode="json")
                    if proposta is not None
                    else None
                ),
                "trilha": resultado["trilha"]
            }

            # Guardar somente as interações concluídas.
            st.session_state.mensagens.append(mensagem_usuario)
            st.session_state.mensagens.append(mensagem_assistente)

            st.rerun()

    except Exception as erro:
        st.error(
            "Não foi possível concluir a solicitação. "
            "Tente novamente."
        )

        with st.expander("Detalhes do erro"):
            st.exception(erro)