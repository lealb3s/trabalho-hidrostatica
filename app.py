#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
app.py
======================================================================
INTERFACE WEB (Streamlit) PARA AS CURVAS HIDROSTATICAS DO 320K VLCC

Este app NAO recalcula nada por conta propria: ele importa e reutiliza
TODO o motor de calculo ja validado em "hydrostatic_curves.py"
(funcoes compute_hydrostatics, buttock_height, dados de offsets, etc.).
Assim, os numeros mostrados aqui sao exatamente os mesmos da versao de
linha de comando -- a interface so adiciona interatividade (sliders,
seletores, tabelas e graficos).

COMO EXECUTAR:
    pip install -r requirements.txt
    streamlit run app.py

O navegador abre automaticamente em http://localhost:8501
======================================================================
"""

import io
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import streamlit as st

# ----------------------------------------------------------------------
# Importa o motor de calculo do projeto (mesmo arquivo de antes).
# Tudo o que o app precisa ja existe la dentro -- nao duplicamos formulas.
# ----------------------------------------------------------------------
import hydrostatic_curves as hc


# ======================================================================
# CONFIGURACAO DA PAGINA
# ======================================================================
st.set_page_config(
    page_title="Curvas Hidrostaticas - 320K VLCC",
    page_icon="🚢",
    layout="wide",
)


# ======================================================================
# CALCULO DA TABELA (com cache para nao recalcular a cada interacao)
# ======================================================================
@st.cache_data
def calcular_tabela():
    """
    Monta a tabela hidrostatica completa chamando compute_hydrostatics
    para cada calado de hc.DRAFTS. Retorna um DataFrame indexado pelo
    calado moldado (m). O resultado fica em cache: so e calculado uma
    vez por sessao, tornando o app instantaneo nas interacoes seguintes.
    """
    linhas = [hc.compute_hydrostatics(float(T)) for T in hc.DRAFTS]
    df = pd.DataFrame(linhas).set_index("Draft_mld")
    return df


def interpolar_em_calado(df, T):
    """
    Interpola linearmente todas as grandezas da tabela hidrostatica em um
    calado T arbitrario (escolhido pelo usuario no slider), usando os
    calados ja calculados como pontos de apoio. Retorna um dicionario
    {grandeza: valor}.
    """
    calados = df.index.values.astype(float)
    out = {}
    for col in df.columns:
        serie = df[col].values.astype(float)
        # np.interp ignora NaN mal se houver; tratamos NaN como ausencia
        mask = ~np.isnan(serie)
        if mask.sum() >= 2:
            out[col] = float(np.interp(T, calados[mask], serie[mask]))
        else:
            out[col] = float("nan")
    return out


# ======================================================================
# GRAFICOS (versoes "para tela", retornando a figura do matplotlib)
# ======================================================================
def fig_curva_geral(df, T_destaque=None):
    """Grafico geral de curvas hidrostaticas (mesma logica/escala/cores
    de hc.CURVAS_DEF). Opcionalmente marca o calado selecionado com uma
    linha horizontal tracejada."""
    fig, ax = plt.subplots(figsize=(11, 8))
    T = df.index.values.astype(float)

    for (col, rotulo, fator, deslocamento), cor in zip(hc.CURVAS_DEF, hc.CORES_CURVAS):
        valor = df[col] / fator + deslocamento
        ax.plot(valor, T, "-o", ms=3.5, lw=1.5, color=cor,
                markeredgecolor="white", markeredgewidth=0.4, label=rotulo)

    if T_destaque is not None:
        ax.axhline(T_destaque, color="red", lw=1.4, ls="--", alpha=0.8)
        ax.text(ax.get_xlim()[1], T_destaque, f" T={T_destaque:.1f} m",
                color="red", va="center", fontsize=9)

    ax.set_xlabel("Valor escalonado dos parametros (fator de escala na legenda)")
    ax.set_ylabel("Calado T [m]")
    ax.set_title("Curvas Hidrostaticas - 320K VLCC", fontweight="bold")
    ax.set_ylim(0, 30)
    ax.set_yticks(np.arange(0, 31, 2))
    ax.xaxis.set_major_locator(MultipleLocator(100))
    ax.xaxis.set_minor_locator(MultipleLocator(50))
    ax.grid(True, which="major", linestyle="-", linewidth=0.7, alpha=0.5)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.5, alpha=0.35)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8,
              title="Grandezas (escala)", title_fontsize=9)
    fig.tight_layout()
    return fig


def fig_grandeza_x_calado(df, coluna, titulo, xlabel, T_destaque=None):
    """Grafico individual de uma grandeza x calado, com a cor da curva
    igual a do grafico geral, e marcador no calado selecionado."""
    fig, ax = plt.subplots(figsize=(7, 6))
    T = df.index.values.astype(float)
    cor = hc.COR_POR_COLUNA.get(coluna, "tab:blue")
    ax.plot(df[coluna].values, T, "-o", ms=4, lw=2.0, color=cor,
            markeredgecolor="white", markeredgewidth=0.4)
    if T_destaque is not None:
        v = interpolar_em_calado(df, T_destaque).get(coluna, np.nan)
        if not np.isnan(v):
            ax.plot([v], [T_destaque], "o", ms=11, mfc="none",
                    mec="red", mew=2.0, zorder=5)
            ax.axhline(T_destaque, color="red", lw=1.0, ls="--", alpha=0.6)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Calado T (m)")
    ax.set_title(titulo)
    ax.set_ylim(0, 30)
    ax.set_yticks(np.arange(0, 31, 2))
    ax.grid(True, linestyle=":", alpha=0.5)
    fig.tight_layout()
    return fig


def fig_body_plan(T_destaque=None):
    """Body Plan (balizas). Se T_destaque for dado, desenha a linha de
    agua correspondente (horizontal) sobre as balizas."""
    fig, ax = plt.subplots(figsize=(7, 6.5))
    cor = "#2b4c7e"
    for lab in hc.STATION_LABELS:
        num = hc.STATION_NUMBERS[hc.STATION_LABELS.index(lab)]
        y = hc.OFFSETS_M[lab]
        y_plot = -y if num <= 10 else y
        ax.plot(y_plot, hc.Z_GRID, "-", color=cor, lw=1.2, alpha=0.9)
    ax.axvline(0, color="black", lw=1.4)
    if T_destaque is not None:
        ax.axhline(T_destaque, color="red", lw=1.4, ls="--", alpha=0.8)
        ax.text(-hc.B / 2, T_destaque, f"WL {T_destaque:.1f} m ",
                color="red", ha="right", va="center", fontsize=9)
    ax.set_xlabel("Meia-boca [m]")
    ax.set_ylabel("Altura z [m]")
    ax.set_title("Body Plan (re <- | -> vante)", fontweight="bold")
    ax.set_xlim(-hc.B / 2 * 1.15, hc.B / 2 * 1.15)
    ax.set_ylim(-1.0, hc.D * 1.04)
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.grid(True, linestyle=":", alpha=0.4)
    fig.tight_layout()
    return fig


def fig_half_breadth():
    """Half-Breadth Plan (linhas de agua) com eixo em estacoes."""
    fig, ax = plt.subplots(figsize=(13, 5))
    st_num = np.array(hc.STATION_NUMBERS, dtype=float)
    cmap = plt.get_cmap("viridis")
    for k, z in enumerate(hc.Z_GRID):
        y_list = [hc.OFFSETS_M[lab][k] for lab in hc.STATION_LABELS]
        ax.plot(st_num, y_list, lw=1.4, color=cmap(z / hc.D), alpha=0.95)
    ax.axvline(10, color="0.4", lw=1.0, ls="--")
    ax.axhline(0, color="black", lw=1.4)
    ax.set_xlabel("(Estacoes / Stations)")
    ax.set_ylabel("Meia-boca [m]")
    ax.set_title("Plano de Linhas de Agua (Half-Breadth Plan)", fontweight="bold")
    ax.set_xlim(-0.6, 20.6)
    ax.set_ylim(0, hc.B / 2 * 1.06)
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, hc.D))
    cb = fig.colorbar(sm, ax=ax, pad=0.01, fraction=0.025)
    cb.set_label("Linha de agua Z [m]")
    fig.tight_layout()
    return fig


def fig_secao_no_calado(T):
    """Desenha a secao transversal a meia-nau preenchida ate o calado T
    (visual didatico de quanto do casco esta submerso)."""
    fig, ax = plt.subplots(figsize=(6, 6.5))
    lab = hc.MIDSHIP_LABEL if hasattr(hc, "MIDSHIP_LABEL") else "10.0"
    y_full = hc.OFFSETS_M[lab]
    # contorno completo da baliza a meia-nau
    ax.plot(y_full, hc.Z_GRID, "-", color="#2b4c7e", lw=1.6)
    ax.plot(-np.array(y_full), hc.Z_GRID, "-", color="#2b4c7e", lw=1.6)
    # parte submersa (ate o calado T) preenchida
    zs, ys = hc.build_subgrid(y_full, T)
    poly_x = np.concatenate([ys, -ys[::-1]])
    poly_z = np.concatenate([zs, zs[::-1]])
    ax.fill(poly_x, poly_z, color="#7fb3e0", alpha=0.6, label="parte submersa")
    ax.axhline(T, color="red", lw=1.4, ls="--", label=f"WL T={T:.1f} m")
    ax.axvline(0, color="black", lw=0.8, alpha=0.5)
    ax.set_xlabel("Meia-boca [m]")
    ax.set_ylabel("Altura z [m]")
    ax.set_title("Secao a meia-nau (parte submersa)", fontweight="bold")
    ax.set_xlim(-hc.B / 2 * 1.1, hc.B / 2 * 1.1)
    ax.set_ylim(0, hc.D * 1.02)
    ax.legend(loc="upper center", fontsize=8)
    ax.grid(True, linestyle=":", alpha=0.4)
    fig.tight_layout()
    return fig


# ======================================================================
# INTERFACE
# ======================================================================
df = calcular_tabela()

st.title("🚢 Curvas Hidrostaticas - 320K VLCC")
st.caption("Term Project 2 - Ship Stability | Interface interativa sobre o "
           "motor de calculo de hydrostatic_curves.py")

# ---- Barra lateral: dados do navio + seletor de calado ----------------
with st.sidebar:
    st.header("Dados principais do navio")
    st.markdown(
        f"""
        | Grandeza | Valor |
        |---|---|
        | LOA | {hc.LOA:.1f} m |
        | LBP | {hc.LBP:.1f} m |
        | Boca (B) | {hc.B:.1f} m |
        | Pontal (D) | {hc.D:.1f} m |
        | Calado projeto (Td) | {hc.TD:.1f} m |
        | Densidade agua mar | {hc.RHO_SW:.3f} t/m³ |
        | Espessura quilha | {hc.KEEL_THK if hasattr(hc,'KEEL_THK') else 0.017:.3f} m |
        """
    )
    st.divider()
    st.header("Calado de interesse")
    T_sel = st.slider("Selecione o calado T (m)", min_value=0.0, max_value=30.0,
                      value=float(hc.TD), step=0.5,
                      help="As grandezas no painel sao interpoladas neste calado.")

# ---- Painel de metricas no calado selecionado -------------------------
vals = interpolar_em_calado(df, T_sel)
st.subheader(f"Valores hidrostaticos interpolados em T = {T_sel:.1f} m")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Volume moldado", f"{vals['Volume_mld']:,.0f} m³")
c2.metric("Deslocamento", f"{vals['Displacement_mld']:,.0f} t")
c3.metric("LCB (da meia-nau)", f"{vals['LCB']:.2f} m")
c4.metric("LCF (da meia-nau)", f"{vals['LCF']:.2f} m")

c5, c6, c7, c8 = st.columns(4)
c5.metric("KB (VCB)", f"{vals['VCB']:.2f} m")
c6.metric("KMT", f"{vals['KMT']:.2f} m")
c7.metric("TPC", f"{vals['TPC']:.2f} t/cm")
c8.metric("MTC", f"{vals['MTC']:,.0f} t·m/cm")

c9, c10, c11, c12 = st.columns(4)
c9.metric("CB", f"{vals['CB']:.4f}")
c10.metric("CWP", f"{vals['CWP']:.4f}")
c11.metric("CM", f"{vals['CM']:.4f}")
c12.metric("CP", f"{vals['CP']:.4f}")

st.divider()

# ---- Abas com os conteudos -------------------------------------------
aba_curvas, aba_individuais, aba_linhas, aba_tabela = st.tabs(
    ["📈 Curvas hidrostaticas", "🔍 Grandeza x calado",
     "📐 Planos de linhas", "📋 Tabela & download"]
)

with aba_curvas:
    st.markdown("Grafico geral de todas as curvas hidrostaticas. A linha "
                "vermelha tracejada marca o calado selecionado na barra lateral.")
    st.pyplot(fig_curva_geral(df, T_destaque=T_sel))
    st.markdown("**Secao a meia-nau** mostrando a parte submersa ate o calado escolhido:")
    st.pyplot(fig_secao_no_calado(T_sel))

with aba_individuais:
    opcoes = {
        "Volume moldado": ("Volume_mld", "Volume (m³)"),
        "Deslocamento": ("Displacement_mld", "Deslocamento (t)"),
        "LCB": ("LCB", "LCB da meia-nau (m)"),
        "LCF": ("LCF", "LCF da meia-nau (m)"),
        "KB (VCB)": ("VCB", "KB (m)"),
        "KMT": ("KMT", "KMT (m)"),
        "KML": ("KML", "KML (m)"),
        "BMT": ("BMT", "BMT (m)"),
        "BML": ("BML", "BML (m)"),
        "MTC": ("MTC", "MTC (t·m/cm)"),
        "TPC": ("TPC", "TPC (t/cm)"),
        "Awp": ("Awp", "Area do plano de flutuacao (m²)"),
        "WSA": ("WSA", "Area molhada (m²)"),
        "CB": ("CB", "CB"),
        "CWP": ("CWP", "CWP"),
        "CM": ("CM", "CM"),
        "CP": ("CP", "CP"),
    }
    escolha = st.selectbox("Escolha a grandeza:", list(opcoes.keys()))
    coluna, xlabel = opcoes[escolha]
    st.pyplot(fig_grandeza_x_calado(df, coluna, f"{escolha} x Calado",
                                    xlabel, T_destaque=T_sel))

with aba_linhas:
    st.markdown("**Body Plan** (a linha de agua do calado selecionado aparece em vermelho):")
    st.pyplot(fig_body_plan(T_destaque=T_sel))
    st.markdown("**Plano de Linhas de Agua (Half-Breadth Plan):**")
    st.pyplot(fig_half_breadth())

with aba_tabela:
    st.markdown("Tabela hidrostatica completa (todos os calados calculados).")

    # Aplica destaque de cor nos cabecalhos das colunas que tem curva
    # (mesmas cores do grafico geral), quando possivel.
    df_show = df.copy()
    st.dataframe(df_show.style.format(precision=3), use_container_width=True)

    # Botoes de download (CSV e XLSX) gerados em memoria
    csv_bytes = df.to_csv(float_format="%.4f").encode("utf-8")
    st.download_button("⬇️ Baixar CSV", data=csv_bytes,
                       file_name="tabela_hidrostatica.csv", mime="text/csv")

    xls_buffer = io.BytesIO()
    with pd.ExcelWriter(xls_buffer, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Hidrostatica")
    st.download_button("⬇️ Baixar XLSX", data=xls_buffer.getvalue(),
                       file_name="tabela_hidrostatica.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

st.caption("Resultados identicos aos de hydrostatic_curves.py (mesmo motor de calculo).")