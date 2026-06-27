#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hydrostatic_curves.py
======================================================================
TERM PROJECT 2 - SHIP STABILITY (Naval Architectural Calculation)
Myung-Il Roh - Seoul National University

Programa em Python (unico arquivo, sem app/Streamlit/Flask) que:
  1) Le a tabela de offsets do 320K VLCC (extraida literalmente do PDF
     do trabalho, convertida de mm para m);
  2) Calcula a tabela hidrostatica completa para os calados solicitados,
     usando integracao numerica (Trapezio, Simpson 1/3 e Simpson 3/8,
     escolhidos automaticamente conforme o espacamento da malha);
  3) Plota as curvas hidrostaticas e as linhas de forma (Body Plan,
     Half-Breadth Plan, Sheer Plan);
  4) Exporta a tabela hidrostatica em CSV e XLSX;
  5) Valida os resultados contra a tabela de referencia apresentada
     no PDF, calculando erro absoluto e percentual.

Bibliotecas utilizadas: NumPy, Pandas e Matplotlib (somente).

Para executar:
    python hydrostatic_curves.py
ou no Google Colab (apos enviar este arquivo / colar o conteudo).
======================================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # backend nao interativo (gera apenas arquivos PNG)
import matplotlib.pyplot as plt

# ======================================================================
# 0. CONFIGURACAO GERAL E PASTA DE SAIDA
# ======================================================================
OUTDIR = "saida_hidrostatica"
os.makedirs(OUTDIR, exist_ok=True)

# ----------------------------------------------------------------------
# ESTILO GLOBAL DOS GRAFICOS
# Define padroes esteticos aplicados a TODAS as figuras geradas, para
# que fiquem bonitas, legiveis e consistentes entre si: fonte um pouco
# maior, grade discreta, fundo branco, resolucao alta e bordas marcadas.
# ----------------------------------------------------------------------
plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 200,                 # PNGs salvos em alta resolucao (mais nitidos)
    "savefig.bbox": "tight",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.labelsize": 12,
    "axes.linewidth": 1.2,              # moldura dos eixos mais marcada
    "axes.edgecolor": "#222222",
    "axes.grid": True,
    "grid.color": "#b8b8b8",
    "grid.linestyle": ":",
    "grid.linewidth": 0.7,
    "grid.alpha": 0.6,
    "legend.fontsize": 9,
    "legend.frameon": True,
    "legend.framealpha": 0.95,
    "legend.edgecolor": "#888888",
    "lines.antialiased": True,
    "xtick.color": "#222222",
    "ytick.color": "#222222",
})


# ======================================================================
# 1. DADOS PRINCIPAIS DO NAVIO (320K VLCC) - extraidos do PDF
# ======================================================================
LOA = 332.8          # comprimento total (m)
LBP = 320.0          # comprimento entre perpendiculares (m)
B   = 60.0           # boca moldada (m)
D   = 30.0           # pontal moldado (m)
TD  = 20.0           # calado de projeto (m)
RHO_SW = 1.025       # densidade da agua do mar (ton/m^3)
KEEL_THK = 0.017     # espessura da chapa de quilha (m)

STATION_SPACING = LBP / 20.0     # espacamento padrao entre estacoes = 16.0 m
X_MID = LBP / 2.0                 # posicao da secao a meia-nau, a partir da A.P. (m)

# ----------------------------------------------------------------------
# Grade de linhas de agua (Z), em metros, a partir da linha de base
# (Bottom Line). Mesma ordem das colunas da tabela de offsets do PDF.
# Observe que o espacamento e 1.0 m entre 0 e 16 m, e passa a ser 2.0 m
# entre 16 e 30 m (exatamente como no PDF).
# ----------------------------------------------------------------------
Z_GRID = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
          18, 20, 22, 24, 26, 28, 30]

# ----------------------------------------------------------------------
# Estacoes (STA. NO.) e respectiva posicao longitudinal X (m), medida a
# partir da perpendicular de re (A.P. = estacao 0).
#   X = numero_da_estacao * (LBP/20)
#
# Por que existem estacoes ALEM da A.P. (-0.333, -0.166) e ALEM da F.P.
# (20.43)? Porque a "Offsets Table of a 320K VLCC" do PDF (paginas 3-4)
# fornece explicitamente meias-bocas nao-nulas nessas estacoes -- elas
# correspondem ao painel de popa (popa/transom, que se projeta um pouco
# para tras da A.P.) e ao painel de proa/roda de proa (que se projeta um
# pouco para frente da F.P.). Ou seja, A.P. e F.P. sao apenas as
# PERPENDICULARES de referencia (usadas para definir LBP = 320 m), e nao
# os limites fisicos reais do casco moldado. Se essas estacoes fossem
# descartadas, o volume e a area de linha de agua calculados ficariam
# sistematicamente subestimados nas extremidades, pois parte do casco
# real (a popa/transom e a roda de proa) deixaria de ser integrada.
# Por isso elas sao mantidas exatamente como fornecidas no PDF, sem
# inventar dados, respeitando o espacamento (nao-uniforme) real entre
# estacoes.
# ----------------------------------------------------------------------
STATION_NUMBERS = [-0.333, -0.166, 0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
                    7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
                    17.0, 18.0, 18.5, 19.0, 19.5, 20.0, 20.43]

STATION_LABELS = ["-0.333", "-0.166", "A.P.(0)", "0.5", "1.0", "2.0", "3.0",
                   "4.0", "5.0", "6.0", "7.0", "8.0", "9.0", "10.0", "11.0",
                   "12.0", "13.0", "14.0", "15.0", "16.0", "17.0", "18.0",
                   "18.5", "19.0", "19.5", "F.P.(20)", "20.43"]

STATION_X = {lab: num * STATION_SPACING
             for lab, num in zip(STATION_LABELS, STATION_NUMBERS)}

MIDSHIP_LABEL = "10.0"   # estacao a meia-nau (estacao 10 de 20)

# ----------------------------------------------------------------------
# TABELA DE OFFSETS (meia-boca, em mm) -- EXATAMENTE como apresentada no
# PDF do trabalho (paginas 3 e 4: "Offsets Table of a 320K VLCC").
# Cada linha segue a mesma ordem de colunas de Z_GRID.
# ----------------------------------------------------------------------
OFFSETS_MM = {
"-0.333":  [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2468,5714,7779,9170,10121,10756,11004],
"-0.166":  [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4375,7391,9426,10762,11695,12342,12624],
"A.P.(0)": [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1063,6079,9047,11017,12341,13189,13760,14172],
"0.5":     [0,0,0,0,0,764,816,0,0,0,0,0,0,168,1837,4321,6952,10875,13560,15298,16576,17417,17959,18398],
"1.0":     [0,2406,3206,3565,3678,3619,3402,3131,2968,3001,3252,3827,4814,6329,8317,10380,12274,15329,17636,19272,20392,21158,21671,22002],
"2.0":     [1980,6807,8190,9059,9678,10145,10541,10954,11466,12154,13093,14276,15597,16949,18245,19424,20493,22378,23898,25016,25802,26361,26692,26880],
"3.0":     [6384,11580,13364,14648,15724,16688,17598,18500,19402,20311,21228,22144,23039,23885,24666,25372,26001,27044,27847,28406,28760,28996,29154,29256],
"4.0":     [11520,16453,18528,20045,21296,22410,23415,24310,25106,25826,26471,27039,27541,27979,28357,28679,28953,29381,29669,29822,29894,29940,29979,30000],
"5.0":     [16608,21227,23138,24564,25705,26633,27391,28007,28502,28896,29207,29450,29636,29775,29875,29944,29984,30000,30000,30000,30000,30000,30000,30000],
"6.0":     [21264,25255,26779,27784,28512,29058,29454,29723,29891,29978,30000,30000,30000,30000,30000,30000,30000,30000,30000,30000,30000,30000,30000,30000],
"7.0":     [24816,27985,28981,29547,29861,29990,30000,30000,30000,30000,30000,30000,30000,30000,30000,30000,30000,30000,30000,30000,30000,30000,30000,30000],
"8.0":     [26736,29221,29818,29997,30000,30000,30000,30000,30000,30000,30000,30000,30000,30000,30000,30000,30000,30000,30000,30000,30000,30000,30000,30000],
"9.0":     [27400,30000,30000,30000,30000,30000,30000,30000,30000,30000,30000,30000,30000,30000,30000,30000,30000,30000,30000,30000,30000,30000,30000,30000],
"10.0":    [27400,30000,30000,30000,30000,30000,30000,30000,30000,30000,30000,30000,30000,30000,30000,30000,30000,30000,30000,30000,30000,30000,30000,30000],
"11.0":    [27400,30000,30000,30000,30000,30000,30000,30000,30000,30000,30000,30000,30000,30000,30000,30000,30000,30000,30000,30000,30000,30000,30000,30000],
"12.0":    [27400,30000,30000,30000,30000,30000,30000,30000,30000,30000,30000,30000,30000,30000,30000,30000,30000,30000,30000,30000,30000,30000,30000,30000],
"13.0":    [27400,30000,30000,30000,30000,30000,30000,30000,30000,30000,30000,30000,30000,30000,30000,30000,30000,30000,30000,30000,30000,30000,30000,30000],
"14.0":    [27400,30000,30000,30000,30000,30000,30000,30000,30000,30000,30000,30000,30000,30000,30000,30000,30000,30000,30000,30000,30000,30000,30000,30000],
"15.0":    [27400,29449,29930,30000,30000,30000,30000,30000,30000,30000,30000,30000,30000,30000,30000,30000,30000,30000,30000,30000,30000,30000,30000,30000],
"16.0":    [25788,28535,29419,29851,29975,30000,30000,30000,30000,30000,30000,30000,30000,30000,30000,30000,30000,30000,30000,30000,30000,30000,30000,30000],
"17.0":    [21972,26146,27478,28336,28935,29340,29586,29706,29754,29760,29760,29760,29760,29760,29760,29760,29760,29760,29760,29760,29760,29760,29760,29760],
"18.0":    [15192,21267,23221,24483,25393,26105,26632,26983,27227,27368,27389,27362,27359,27361,27359,27360,27360,27360,27360,27360,27371,27459,27619,27852],
"18.5":    [10644,17313,19482,20938,22038,22879,23521,23994,24314,24542,24696,24786,24815,24817,24815,24816,24816,24816,24816,24861,25029,25297,25641,26040],
"19.0":    [5880,12220,14539,16152,17372,18277,19013,19626,20082,20421,20667,20831,20916,20934,20899,20832,20772,20755,20868,21132,21485,21945,22541,23280],
"19.5":    [1860,6613,8767,10271,11400,12291,13009,13589,14048,14398,14640,14775,14812,14760,14627,14460,14348,14347,14640,15180,15908,16798,17821,18960],
"F.P.(20)":[0,1875,3102,4014,4751,5371,5883,6295,6615,6850,7005,7067,7023,6868,6600,6190,5568,3172,0,1872,5178,8178,10707,12960],
"20.43":   [0,0,0,0,0,0,326,562,678,754,800,822,817,741,524,0,0,0,0,0,0,0,0,3192],
}

# Conversao de mm para m (todos os calculos hidrostaticos sao feitos em m)
OFFSETS_M = {k: np.array(v, dtype=float) / 1000.0 for k, v in OFFSETS_MM.items()}

# Calados (T, em metros) a serem calculados, conforme solicitado
DRAFTS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
          20, 22, 24, 26, 28, 30]


# ======================================================================
# 2. FUNCOES DE INTEGRACAO NUMERICA (Trapezio, Simpson 1/3, Simpson 3/8)
# ======================================================================
def trapz_rule(x, y):
    """
    Regra do Trapezio (generica, aceita espacamento NAO uniforme):
        Integral = soma_i [ (y_i + y_{i+1})/2 * (x_{i+1} - x_i) ]
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return float(np.sum((y[1:] + y[:-1]) / 2.0 * (x[1:] - x[:-1])))


def simpson13_rule(x, y):
    """
    Regra de Simpson 1/3 (exige espacamento UNIFORME h e numero PAR de
    intervalos, ou seja, numero IMPAR de ordenadas):
        Integral = h/3 * [ y0 + y_n + 4*sum(impares) + 2*sum(pares internos) ]
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x) - 1                       # numero de intervalos
    h = (x[-1] - x[0]) / n
    soma_impar = np.sum(y[1:-1:2])       # ordenadas de indice impar (1,3,5,...)
    soma_par = np.sum(y[2:-1:2])         # ordenadas de indice par interno (2,4,6,...)
    return float(h / 3.0 * (y[0] + y[-1] + 4.0 * soma_impar + 2.0 * soma_par))


def simpson38_rule(x, y):
    """
    Regra de Simpson 3/8 (exige espacamento UNIFORME h e numero de
    intervalos MULTIPLO de 3):
        Para cada bloco de 3 intervalos (4 ordenadas):
        Integral_bloco = 3h/8 * (y0 + 3y1 + 3y2 + y3)
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x) - 1
    h = (x[-1] - x[0]) / n
    total = 0.0
    i = 0
    while i + 3 <= n:
        total += 3.0 * h / 8.0 * (y[i] + 3 * y[i + 1] + 3 * y[i + 2] + y[i + 3])
        i += 3
    return float(total)


def _runs_de_espacamento_uniforme(x, tol=1e-6):
    """
    Identifica sub-trechos (runs) de x onde o espacamento entre pontos
    consecutivos e constante (dentro de uma tolerancia). Retorna lista
    de tuplas (indice_inicial, indice_final) de cada trecho uniforme.
    """
    n = len(x)
    runs = []
    i = 0
    while i < n - 1:
        dx0 = x[i + 1] - x[i]
        j = i + 1
        while j < n - 1 and abs((x[j + 1] - x[j]) - dx0) <= tol * max(1.0, abs(dx0)):
            j += 1
        runs.append((i, j))
        i = j
    return runs


def integrate_general(x, y):
    """
    FUNCAO GERAL DE INTEGRACAO NUMERICA.

    Estrategia: a malha (x,y) e dividida automaticamente em sub-trechos
    de espacamento uniforme. Em cada sub-trecho, o metodo mais preciso
    disponivel e escolhido automaticamente:
        - numero de intervalos PAR (>=2)         -> Simpson 1/3
        - numero de intervalos MULTIPLO de 3      -> Simpson 3/8
        - caso contrario (ex.: 1 intervalo, ou
          espacamento NAO uniforme)               -> Trapezio
    Os resultados parciais sao somados, fornecendo a integral total.
    Isso permite tratar tanto a malha das linhas de agua (parcialmente
    uniforme) quanto a malha das estacoes (claramente NAO uniforme,
    pois o offsets table do VLCC tem estacoes fracionarias e estacoes
    alem da A.P./F.P.).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    if n < 2:
        return 0.0
    if n == 2:
        return trapz_rule(x, y)

    total = 0.0
    for (i, j) in _runs_de_espacamento_uniforme(x):
        seg_x = x[i:j + 1]
        seg_y = y[i:j + 1]
        m = len(seg_x) - 1     # numero de intervalos no sub-trecho
        if m <= 0:
            continue
        elif m % 2 == 0:
            total += simpson13_rule(seg_x, seg_y)
        elif m % 3 == 0:
            total += simpson38_rule(seg_x, seg_y)
        else:
            total += trapz_rule(seg_x, seg_y)
    return total


# ======================================================================
# 3. PREPARACAO DA TABELA DE OFFSETS / INTERPOLACAO LINEAR
# ======================================================================
def build_subgrid(row_vals_m, T):
    """
    Constroi a sub-malha (z, y) de uma estacao, do fundo (z=0) at o
    calado T. Se T nao coincidir com nenhuma linha de agua da tabela
    original (por exemplo T = 17 m, que nao existe na tabela -- ela
    salta de 16 para 18 m), a meia-boca em z=T e obtida por
    INTERPOLACAO LINEAR entre as duas linhas de agua vizinhas, exatamente
    como solicitado no enunciado ("se necessario, use interpolacao
    linear entre os dados fornecidos").
    """
    zs, ys = [], []
    for z, y in zip(Z_GRID, row_vals_m):
        if z <= T + 1e-9:
            zs.append(z)
            ys.append(y)
        else:
            if abs(zs[-1] - T) > 1e-9:
                z0, y0 = zs[-1], ys[-1]
                z1, y1 = z, y
                yT = y0 + (y1 - y0) * (T - z0) / (z1 - z0)
                zs.append(T)
                ys.append(yT)
            break
    return np.array(zs, dtype=float), np.array(ys, dtype=float)


def half_breadth_at(row_vals_m, T):
    """Retorna a meia-boca (m) de uma estacao exatamente no calado T."""
    zs, ys = build_subgrid(row_vals_m, T)
    return ys[-1]


# ======================================================================
# 4. CALCULO DE AREA SECCIONAL E 1o MOMENTO VERTICAL (POR ESTACAO)
# ======================================================================
def compute_section(row_vals_m, T):
    """
    Para uma dada estacao e um calado T, calcula:
        S = area da secao transversal submersa (m^2)
            S = 2 * integral_0^T  y(z) dz          (fator 2: bombordo+boreste)
        M = 1o momento da area em relacao a linha de base (m^3)
            M = 2 * integral_0^T  y(z)*z dz
    O valor S/Volume (apos integracao longitudinal) fornece o KB (VCB).
    """
    zs, ys = build_subgrid(row_vals_m, T)
    if len(zs) < 2:
        return 0.0, 0.0
    S = 2.0 * integrate_general(zs, ys)
    M = 2.0 * integrate_general(zs, ys * zs)
    return S, M


# ======================================================================
# 5. CALCULO HIDROSTATICO COMPLETO PARA UM DADO CALADO T
# ======================================================================
def compute_hydrostatics(T):
    """
    Calcula todas as grandezas hidrostaticas solicitadas para o calado
    moldado T (m), por meio de:
      (a) integracao VERTICAL (em z) -> area seccional S(x) e momento M(x)
          de cada estacao;
      (b) integracao LONGITUDINAL (em x) das areas/momentos das estacoes
          -> Volume, LCB, VCB(KB);
      (c) propriedades do plano de flutuacao na cota z=T (meia-boca
          exatamente no calado) -> Awp, LCF, momentos de inercia
          transversal/longitudinal -> BMT, BML, KMT, KML, MTC, TPC;
      (d) area molhada aproximada (WSA) via comprimento de arco do
          contorno submerso de cada estacao;
      (e) coeficientes de forma CB, CWP, CM, CP.
    """
    x_arr, S_arr, M_arr, y_arr = [], [], [], []

    for lab in STATION_LABELS:
        row = OFFSETS_M[lab]
        S, M = compute_section(row, T)
        yT = half_breadth_at(row, T)
        x_arr.append(STATION_X[lab])
        S_arr.append(S)
        M_arr.append(M)
        y_arr.append(yT)

    x_arr = np.array(x_arr)
    S_arr = np.array(S_arr)
    M_arr = np.array(M_arr)
    y_arr = np.array(y_arr)

    # ----------------------------------------------------------------------
    # CASO ESPECIAL: calado T = 0 (navio NAO imerso / sem flutuacao real)
    # ----------------------------------------------------------------------
    # Em T = 0, "y_arr" (meia-boca na cota z=T=0) e simplesmente a meia-boca
    # do FUNDO CHATO de cada estacao (a linha "Bottom Line" da tabela de
    # offsets, que e diferente de zero -- por exemplo, ~27.4 m a meia-nau).
    # Se essa largura fosse usada diretamente nas formulas gerais de Awp,
    # TPC, CWP, CB, CM e CP, essas grandezas resultariam em valores
    # POSITIVOS mesmo sem nenhum volume submerso, o que NAO tem sentido
    # fisico (um navio com calado nulo nao tem volume deslocado, nao tem
    # plano de flutuacao real e, portanto, nao tem TPC/coeficientes de
    # forma definidos). Por isso, esse caso e tratado explicitamente aqui,
    # retornando uma tabela hidrostatica com todas essas grandezas iguais
    # a zero, sem alterar o restante do algoritmo (que permanece valido
    # para T > 0).
    if T <= 1e-9:
        nan = float("nan")
        return {
            "Draft_mld": 0.0,
            "Draft_ext": KEEL_THK,     # apenas a espessura da quilha (sem volume submerso)
            "Volume_mld": 0.0,
            "Volume_ext": 0.0,
            "Displacement_mld": 0.0,
            "Displacement_ext": 0.0,
            # LCB e LCF sao formas indeterminadas (0/0: momento e area/volume
            # tendem a zero JUNTOS quando T->0). O limite fisico real e um
            # valor FINITO e NAO-NULO (o centroide do fundo chato da
            # estacao), e nao zero -- por isso usa-se NaN (em vez de 0.0)
            # para nao criar um salto artificial nos graficos entre T=0 e
            # o primeiro calado calculado.
            "LCB": nan,
            "LCF": nan,
            "VCB": 0.0,   # KB tende corretamente a 0 quando T->0 (sem singularidade)
            "KBT": 0.0,
            # BMT, BML, KMT, KML e MTC SAO REALMENTE SINGULARES em T=0
            # (BMT = I_T/Volume e BML = I_L/Volume, com Volume->0 mas
            # I_T, I_L tendendo a valores finitos => BMT, BML -> infinito).
            # Portanto 0.0 seria fisicamente ERRADO aqui (o valor real
            # diverge, nao se anula); usa-se NaN para deixar essa
            # indeterminacao explicita e evitar o "bico" artificial que
            # aparecia antes nos graficos de KMT/KML/MTC.
            "BMT": nan,
            "KMT": nan,
            "KBL": 0.0,
            "BML": nan,
            "KML": nan,
            "MTC": nan,
            "TPC": 0.0,                # sem plano de flutuacao real -> TPC = 0
            "WSA": 0.0,
            "Awp": 0.0,                # sem plano de flutuacao real -> Awp = 0
            "CB": 0.0,                 # sem volume submerso -> coeficientes de forma = 0
            "CWP": 0.0,
            "CM": 0.0,
            "CP": 0.0,
        }

    # ---------------- (b) Integracao longitudinal -> Volume, LCB, VCB ----
    # Volume deslocado moldado (m^3): Volume = integral_AP^FP S(x) dx
    Volume_mld = integrate_general(x_arr, S_arr)

    # ------------------------------------------------------------------
    # LCB (Longitudinal Center of Buoyancy) -- ORIGEM E SINAL:
    #   - "LCB_AP" e calculado em relacao a A.P. (estacao 0, X = 0 m),
    #     com X crescendo no sentido da A.P. para a F.P. (X = LBP = 320 m
    #     na F.P.), exatamente como definido em STATION_X / X_MID.
    #   - LCB_AP = (integral S(x)*x dx) / Volume  ->  centroide longitudinal
    #     da distribuicao de area seccional submersa, medido a partir da A.P.
    #   - "LCB_mid" (o valor efetivamente reportado na tabela final) e
    #     LCB_AP - X_MID, ou seja, a distancia do centro de carena em
    #     relacao a MEIA-NAU (estacao 10).
    #   - CONVENCAO DE SINAL adotada: LCB_mid > 0  => centro de carena a
    #     FRENTE da meia-nau (no sentido da F.P. / proa);
    #     LCB_mid < 0  => centro de carena a RE da meia-nau (no sentido
    #     da A.P. / popa). Esta e a mesma convencao usada na tabela de
    #     referencia do PDF (ver item 3 da validacao, mais abaixo).
    # ------------------------------------------------------------------
    if Volume_mld > 1e-9:
        Mx = integrate_general(x_arr, S_arr * x_arr)
        LCB_AP = Mx / Volume_mld
        # VCB = KB (a partir da linha de base): KB = (integral M dx)/Volume
        Mz = integrate_general(x_arr, M_arr)
        KB = Mz / Volume_mld
    else:
        LCB_AP, KB = X_MID, 0.0

    LCB_mid = LCB_AP - X_MID   # LCB medido a partir da meia-nau (m); sinal: ver acima

    # ---------------- (c) Plano de flutuacao na cota T --------------------
    # Area do plano de flutuacao (m^2): Awp = integral 2*y(x,T) dx
    Awp = integrate_general(x_arr, 2.0 * y_arr)

    # ------------------------------------------------------------------
    # LCF (Longitudinal Center of Flotation) -- ORIGEM E SINAL:
    #   - Mesma origem (A.P., X = 0) e mesmo sentido positivo (A.P. -> F.P.)
    #     usados para o LCB acima.
    #   - "LCF_AP" e o centroide longitudinal da AREA DO PLANO DE FLUTUACAO
    #     na cota z = T (e nao do volume submerso, como no LCB):
    #     LCF_AP = (integral 2*y(x,T)*x dx) / Awp.
    #   - "LCF_mid" = LCF_AP - X_MID, reportado na tabela final, com a
    #     MESMA CONVENCAO DE SINAL do LCB_mid: positivo => a frente da
    #     meia-nau (sentido da F.P.); negativo => a re da meia-nau
    #     (sentido da A.P.).
    # ------------------------------------------------------------------
    if Awp > 1e-9:
        Mx_wp = integrate_general(x_arr, 2.0 * y_arr * x_arr)
        LCF_AP = Mx_wp / Awp
    else:
        LCF_AP = X_MID
    LCF_mid = LCF_AP - X_MID   # LCF medido a partir da meia-nau (m); sinal: ver acima

    # Momento de inercia TRANSVERSAL do plano de flutuacao em relacao a
    # linha de centro (usado para BMT):  I_T = integral (2/3)*y^3 dx
    I_T = integrate_general(x_arr, (2.0 / 3.0) * y_arr ** 3)
    # Momento de inercia LONGITUDINAL do plano de flutuacao em relacao a
    # um eixo transversal passando pelo LCF (usado para BML):
    #   I_L = integral 2*y*(x - x_F)^2 dx
    I_L = integrate_general(x_arr, 2.0 * y_arr * (x_arr - LCF_AP) ** 2)

    BMT = I_T / Volume_mld if Volume_mld > 1e-9 else 0.0
    BML = I_L / Volume_mld if Volume_mld > 1e-9 else 0.0
    KMT = KB + BMT
    KML = KB + BML
    KBT = KB     # KB usado no calculo transversal (mesmo ponto fisico)
    KBL = KB     # KB usado no calculo longitudinal (mesmo ponto fisico)

    Displacement_mld = Volume_mld * RHO_SW

    # TPC = toneladas por centimetro de imersao = (Awp * rho)/100
    TPC = Awp * RHO_SW / 100.0
    # MTC = momento para alterar o trim em 1 cm = (Desloc * BML)/(100*LBP)
    # (formula classica de tabelas hidrostaticas, independente do KG)
    MTC = Displacement_mld * BML / (100.0 * LBP)

    # ---------------- (d) Area molhada (WSA) - aproximacao por girth -----
    # Para cada estacao: comprimento do contorno submerso (de bombordo a
    # boreste), passando pelo fundo chato (2*y0, espessura desprezavel) e
    # subindo pela curva da boca ate a linha de agua T (comprimento de
    # arco). WSA = integral do "girth" ao longo do comprimento do navio.
    girths = []
    for lab in STATION_LABELS:
        row = OFFSETS_M[lab]
        zs, ys = build_subgrid(row, T)
        if len(zs) < 2:
            girths.append(0.0)
            continue
        arco_lateral = float(np.sum(np.sqrt(np.diff(zs) ** 2 + np.diff(ys) ** 2)))
        y0 = ys[0]                       # meia-boca no fundo (z=0)
        girth_total = 2.0 * (y0 + arco_lateral)   # fundo + os dois bordos
        girths.append(girth_total)
    girths = np.array(girths)
    WSA = integrate_general(x_arr, girths)

    # ---------------- (e) Coeficientes de forma --------------------------
    B_T = float(np.max(2.0 * y_arr)) if np.max(y_arr) > 0 else 0.0   # boca na linha de agua T
    row_mid = OFFSETS_M[MIDSHIP_LABEL]
    Am, _ = compute_section(row_mid, T)    # area seccional a meia-nau

    if T > 1e-9 and B_T > 1e-9:
        CB = Volume_mld / (LBP * B_T * T)
        CM = Am / (B_T * T)
    else:
        CB, CM = 0.0, 0.0
    CWP = Awp / (LBP * B_T) if B_T > 1e-9 else 0.0
    CP = Volume_mld / (Am * LBP) if Am > 1e-9 else 0.0

    # ---------------- Volume/Deslocamento EXTREMOS (casco extremo) -------
    # So a espessura da chapa de quilha foi fornecida no PDF (nenhuma
    # espessura de chapeamento do costado foi dada). Adota-se, portanto,
    # a hipotese simplificadora de que o calado extremo adiciona uma fina
    # camada de espessura igual a chapa de quilha por baixo da linha de
    # base moldada, contribuindo um volume extra aproximado por:
    #     dV = integral [2*y(x, z=0)] dx * KEEL_THK
    # (area do fundo chato vezes a espessura da quilha)
    y0_arr = np.array([OFFSETS_M[lab][0] for lab in STATION_LABELS])
    dV = integrate_general(x_arr, 2.0 * y0_arr) * KEEL_THK
    Volume_ext = Volume_mld + dV
    Displacement_ext = Volume_ext * RHO_SW
    T_ext = T + KEEL_THK

    return {
        "Draft_mld": T,
        "Draft_ext": T_ext,
        "Volume_mld": Volume_mld,
        "Volume_ext": Volume_ext,
        "Displacement_mld": Displacement_mld,
        "Displacement_ext": Displacement_ext,
        "LCB": LCB_mid,
        "LCF": LCF_mid,
        "VCB": KB,
        "KBT": KBT,
        "BMT": BMT,
        "KMT": KMT,
        "KBL": KBL,
        "BML": BML,
        "KML": KML,
        "MTC": MTC,
        "TPC": TPC,
        "WSA": WSA,
        "Awp": Awp,
        "CB": CB,
        "CWP": CWP,
        "CM": CM,
        "CP": CP,
    }


# ======================================================================
# 6. MONTAGEM DA TABELA HIDROSTATICA PARA TODOS OS CALADOS SOLICITADOS
# ======================================================================
def montar_tabela_hidrostatica():
    print("=" * 78)
    print("CALCULANDO TABELA HIDROSTATICA DO 320K VLCC")
    print("=" * 78)
    linhas = []
    for T in DRAFTS:
        res = compute_hydrostatics(float(T))
        linhas.append(res)
        print(f"  Calado moldado T = {T:5.2f} m  ->  Volume_mld = {res['Volume_mld']:10.1f} m^3 "
              f"| Desloc_mld = {res['Displacement_mld']:10.1f} ton")
    df = pd.DataFrame(linhas)
    df = df.set_index("Draft_mld")
    return df


# ======================================================================
# 7. VALIDACAO COM A TABELA DE REFERENCIA DO PDF
# ======================================================================
# Tabela de referencia "[References] Example of Hydrostatic Tables of a
# 320K VLCC" (pagina 4 do PDF). Os calados de referencia (moldados) NAO
# coincidem exatamente com os calados solicitados neste trabalho; por
# isso, a curva calculada e interpolada nos mesmos calados de referencia
# para permitir a comparacao grandeza a grandeza.
#
# VERIFICACAO DE ORIGEM E SENTIDO PARA LCB e LCF (item importante da
# validacao): a propria tabela do PDF rotula essas linhas como
# "L.C.B. FROM MIDSHIP" e "L.C.F. FROM MIDSHIP", ou seja, ambas medidas
# a partir da meia-nau -- exatamente a mesma referencia usada em
# LCB_mid/LCF_mid (calculadas como LCB_AP - X_MID e LCF_AP - X_MID, ver
# compute_hydrostatics()). Quanto ao SENTIDO (sinal): os valores de
# referencia de LCB e LCF DIMINUEM com o aumento do calado e o LCF passa
# de positivo para negativo proximo de T ~ 22 m; a serie calculada por
# este programa reproduz exatamente o mesmo comportamento (LCB e LCF
# tambem diminuem com o calado, e o LCF tambem muda de sinal por volta do
# mesmo calado). Essa coincidencia de tendencia/sinal confirma que ambas
# as series usam a mesma convencao: positivo = a frente da meia-nau (no
# sentido da F.P.); negativo = a re da meia-nau (no sentido da A.P.).
# Logo, a comparacao numerica feita em validar_resultados() e consistente
# (mesma origem, mesmo sentido), e nao requer nenhuma inversao de sinal.
REF_DRAFT_MLD = np.array([14.983, 15.983, 16.983, 17.983, 18.983, 19.983, 20.983, 21.983])

REF_VALUES = {
    "Volume_mld":       [235335.3, 252293.8, 269404.3, 286658.7, 304044.4, 321546.8, 339154.3, 356853.1],
    "Displacement_mld": [241596.7, 258991.0, 276541.2, 294238.9, 312070.8, 330022.1, 348091.0, 366233.3],  # disp. agua salgada
    "LCB":              [15.790, 13.327, 12.816, 12.268, 11.705, 11.146, 10.602, 10.083],
    "LCF":              [7.680, 6.094, 4.477, 3.015, 1.888, 1.018, 0.365, -0.100],
    "VCB":              [7.727, 8.249, 8.772, 9.296, 9.822, 10.347, 10.874, 11.400],
    "TPC":              [173.0, 174.6, 176.1, 177.6, 178.8, 180.0, 181.0, 181.8],
    "MTC":              [3699.8, 3801.8, 3902.6, 3996.3, 4077.4, 4150.3, 4214.6, 4271.6],
    "KMT":              [27.100, 26.484, 26.006, 25.642, 25.368, 25.171, 25.032, 24.945],
    "KML":              [497.777, 477.985, 460.362, 443.913, 427.922, 412.771, 398.330, 384.638],
    "Awp":              [16882.6, 17034.8, 17183.7, 17322.8, 17445.8, 17557.2, 17655.2, 17741.6],
    "WSA":              [24585.0, 25357.5, 26137.6, 26910.5, 27661.0, 28398.1, 29126.0, 29847.8],
    "CB":               [0.8181, 0.8221, 0.8262, 0.8302, 0.8342, 0.8381, 0.8418, 0.8455],
    "CP":               [0.8208, 0.8247, 0.8286, 0.8325, 0.8364, 0.8401, 0.8438, 0.8474],
    "CWP":              [0.8793, 0.8872, 0.8950, 0.9022, 0.9086, 0.9144, 0.9195, 0.9240],
    "CM":               [0.9967, 0.9969, 0.9971, 0.9973, 0.9974, 0.9975, 0.9977, 0.9978],
}


def validar_resultados(df):
    print()
    print("=" * 78)
    print("VALIDACAO: COMPARACAO COM A TABELA DE REFERENCIA DO PDF")
    print("(curva calculada interpolada nos calados de referencia)")
    print("=" * 78)

    registros = []
    calados_calc = df.index.values.astype(float)

    for grandeza, valores_ref in REF_VALUES.items():
        valores_ref = np.array(valores_ref, dtype=float)
        serie_calc = df[grandeza].values.astype(float)
        # interpola a curva calculada nos calados de referencia
        valores_calc_interp = np.interp(REF_DRAFT_MLD, calados_calc, serie_calc)
        for Tr, vcalc, vref in zip(REF_DRAFT_MLD, valores_calc_interp, valores_ref):
            erro_abs = vcalc - vref
            erro_pct = 100.0 * erro_abs / vref if abs(vref) > 1e-9 else np.nan
            registros.append({
                "grandeza": grandeza,
                "calado_m": Tr,
                "valor_calculado": vcalc,
                "valor_referencia": vref,
                "erro_absoluto": erro_abs,
                "erro_percentual": erro_pct,
            })

    df_val = pd.DataFrame(registros)

    # Destaca no terminal os erros percentuais maiores que 2%
    print(f"\n{'Grandeza':<18}{'Calado(m)':>10}{'Calculado':>16}{'Referencia':>16}"
          f"{'Erro Abs.':>14}{'Erro %':>10}")
    print("-" * 86)
    for _, row in df_val.iterrows():
        marca = "  <-- ERRO > 2%" if (not np.isnan(row["erro_percentual"])
                                       and abs(row["erro_percentual"]) > 2.0) else ""
        print(f"{row['grandeza']:<18}{row['calado_m']:>10.3f}{row['valor_calculado']:>16.3f}"
              f"{row['valor_referencia']:>16.3f}{row['erro_absoluto']:>14.3f}"
              f"{row['erro_percentual']:>9.2f}%{marca}")

    n_erros_grandes = int((df_val["erro_percentual"].abs() > 2.0).sum())
    print("-" * 86)
    print(f"Total de comparacoes: {len(df_val)} | "
          f"Comparacoes com erro percentual > 2%: {n_erros_grandes}")
    print("NOTA: divergencias sao esperadas pois (i) o navio de referencia do PDF")
    print("foi calculado para calados diferentes (interpolados aqui) e (ii) o WSA e")
    print("o MTC usam formulas/aproximacoes proprias deste programa (girth aproximado")
    print("e formula classica MTC = Desloc*BML/(100*LBP)).")

    df_val.to_csv(os.path.join(OUTDIR, "tabela_validacao.csv"), index=False)
    return df_val


# ======================================================================
# 8. GRAFICOS (CURVAS HIDROSTATICAS E LINHAS DE FORMA)
# ======================================================================
# ----------------------------------------------------------------------
# DEFINICAO CENTRALIZADA DAS CURVAS HIDROSTATICAS
# Cada item: (coluna_no_df, rotulo_legenda, fator_escala, deslocamento).
#   valor_plotado = valor_real / fator + deslocamento
# Esta lista e usada TANTO pelo grafico geral de curvas hidrostaticas
# QUANTO pela tabela hidrostatica colorida -- garantindo que a cor de
# cada grandeza seja EXATAMENTE A MESMA nos dois lugares (curva e tabela).
# ----------------------------------------------------------------------
CURVAS_DEF = [
    ("Volume_mld",       "Volume (m3)  [1:1000]",            1000.0,    0.0),
    ("Displacement_mld", "Deslocamento (ton) [1:1000]+5",    1000.0,    5.0),
    ("LCB",              "LCB da meia-nau (m) [1:0.1]+200",     0.1,  200.0),
    ("LCF",              "LCF da meia-nau (m) [1:0.5]+100",     0.5,  100.0),
    ("VCB",              "VCB (m) [1:0.1]",                     0.1,    0.0),
    ("KMT",              "KMT (m) [1:1]+10",                    1.0,   10.0),
    ("KML",              "KML (m) [1:50]+35",                  50.0,   35.0),
    ("MTC",              "MTC (t.m/cm) [1:20]+90",             20.0,   90.0),
    ("TPC",              "TPC (t/cm) [1:1]+20",                 1.0,   20.0),
    ("Awp",              "Awp (m2) [1:100]+10",               100.0,   10.0),
    ("WSA",              "WSA (m2) [1:100]",                  100.0,    0.0),
    ("CB",               "CB [1:0.005]-5",                     0.005, -5.0),
    ("CWP",              "CWP [1:0.01]",                       0.01,   0.0),
    ("CM",               "CM [1:0.01]",                        0.01,   0.0),
    ("CP",               "CP [1:0.005]+35",                    0.005, 35.0),
]

# Paleta com 1 cor distinta por curva (sem repeticao -> evita o efeito
# de "linhas que se confundem"). A MESMA lista de cores e reutilizada na
# tabela colorida, na ordem de CURVAS_DEF.
_CMAP_CURVAS = plt.get_cmap("tab20")
CORES_CURVAS = [_CMAP_CURVAS(i / 20.0) for i in range(len(CURVAS_DEF))]
# Mapa coluna -> cor, para consulta direta (ex.: na tabela colorida)
COR_POR_COLUNA = {col: CORES_CURVAS[i] for i, (col, *_rest) in enumerate(CURVAS_DEF)}


def plot_curva_geral(df):
    """
    Grafico geral de curvas hidrostaticas no estilo classico de uma
    tabela/grafico hidrostatico de navio (igual ao grafico de referencia
    "[References] Example of Hydrostatic Curves of a 320K VLCC", pag. 5
    do PDF do trabalho): cada grandeza e plotada em uma escala propria,
    indicada na legenda com a notacao "[1:fator]+deslocamento", ou seja:

        valor_plotado = valor_real / fator + deslocamento

    Essa notacao "1:fator" e a mesma usada em desenho tecnico/escalas de
    plantas (ex.: "1:1000" significa que 1 unidade no grafico representa
    1000 unidades reais, isto e, divide-se o valor real por 1000). O
    "+deslocamento" apenas afasta a curva das demais no eixo X, sem
    alterar sua forma, para facilitar a leitura visual quando varias
    curvas ficam proximas.

    O calado (eixo Y) cobre a faixa COMPLETA calculada (0 a 30 m, de 2
    em 2 m), e cada ponto calculado e marcado com um circulo ("o"),
    reproduzindo o aspecto do grafico de referencia.
    """
    fig, ax = plt.subplots(figsize=(14, 9.5))
    T = df.index.values.astype(float)

    # Usa a definicao centralizada CURVAS_DEF + CORES_CURVAS (as MESMAS
    # cores aparecem na tabela hidrostatica colorida).
    for (col, rotulo, fator, deslocamento), cor in zip(CURVAS_DEF, CORES_CURVAS):
        valor_plotado = df[col] / fator + deslocamento
        # NaN em T=0 (LCB, LCF, KMT, KML, MTC -- ver compute_hydrostatics)
        # simplesmente nao geram marcador/segmento nesse ponto.
        ax.plot(valor_plotado, T, "-o", ms=4, lw=1.6, color=cor,
                markeredgecolor="white", markeredgewidth=0.4, label=rotulo)

    ax.set_ylabel("Calado T [m]")
    ax.set_xlabel("Valor escalonado dos parametros (fator de escala na legenda)")
    ax.set_title("Curvas Hidrostaticas - 320K VLCC")

    # ---- Eixo Y: faixa COMPLETA de calados calculados (0 a 30 m) -------
    ax.set_ylim(0, 30)
    ax.set_yticks(np.arange(0, 31, 2))

    # ---- Eixo X: marcacoes a cada 100, limite superior ajustado
    # automaticamente um pouco acima do maior valor escalonado presente,
    # para que nenhuma curva fique cortada -------------------------------
    from matplotlib.ticker import MultipleLocator
    maximos = []
    for col, _rot, fator, deslocamento in CURVAS_DEF:
        v = (df[col] / fator + deslocamento).values
        v = v[~np.isnan(v)]
        if len(v):
            maximos.append(np.nanmax(v))
    x_max_dado = max(maximos) if maximos else 500.0
    x_lim_sup = (int(x_max_dado // 50) + 2) * 50   # proximo multiplo de 50 + margem
    ax.set_xlim(0, x_lim_sup)
    ax.xaxis.set_major_locator(MultipleLocator(100))
    ax.xaxis.set_minor_locator(MultipleLocator(50))
    ax.grid(True, which="major", linestyle="-", linewidth=0.7, alpha=0.5)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.5, alpha=0.35)

    # Legenda posicionada FORA da area do grafico (ao lado direito), para
    # nao sobrepor as curvas.
    ax.legend(loc="center left", bbox_to_anchor=(1.015, 0.5), fontsize=9,
              ncol=1, title="Grandezas (escala)", title_fontsize=10,
              borderaxespad=0.0)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "01_curvas_hidrostaticas_geral.png"))
    plt.close(fig)


def plot_simples(df, coluna, titulo, xlabel, arquivo):
    """
    Grafico individual de uma grandeza hidrostatica x calado, cobrindo a
    faixa COMPLETA de calados calculados (0 a 30 m, marcacoes a cada
    2 m), com marcador em cada ponto calculado. A cor da curva e a MESMA
    usada para essa grandeza no grafico geral e na tabela colorida.
    """
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    T = df.index.values.astype(float)
    cor = COR_POR_COLUNA.get(coluna, "tab:blue")
    ax.plot(df[coluna].values, T, "-o", ms=4, lw=2.0, color=cor,
            markeredgecolor="white", markeredgewidth=0.4)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Calado T (m)")
    ax.set_title(titulo)
    ax.set_ylim(0, 30)
    ax.set_yticks(np.arange(0, 31, 2))
    ax.grid(True, linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, arquivo))
    plt.close(fig)


def plot_body_plan():
    """
    Body Plan (Plano de Balizas): meia-boca espelhada (-B/2 a +B/2, m) vs
    altura Z (m), para cada estacao, todas sobrepostas no mesmo grafico
    (re a esquerda do eixo, vante a direita -- convencao classica em que
    o lado esquerdo do body plan mostra as balizas de re e o lado direito
    as balizas de vante).
    """
    from matplotlib.ticker import MultipleLocator
    fig, ax = plt.subplots(figsize=(9, 7.5))   # mais largo -> balizas com mais respiro
    cor = "#2b4c7e"   # azul, no estilo do plano de linhas classico

    for lab in STATION_LABELS:
        num = STATION_NUMBERS[STATION_LABELS.index(lab)]
        y = OFFSETS_M[lab]
        # Estacoes de re (numero < meia-nau) desenhadas do lado negativo
        # (esquerdo) e estacoes de vante do lado positivo (direito),
        # reproduzindo a convencao "re <- | -> vante" do body plan.
        y_plot = -y if num <= 10 else y
        ax.plot(y_plot, Z_GRID, "-", color=cor, lw=1.3, alpha=0.9)

    ax.axvline(0, color="black", lw=1.4)
    ax.set_xlabel("Meia-boca [m]")
    ax.set_ylabel("Altura z [m]")
    ax.set_title("Plano de Balizas (Body Plan)\nre <- | -> vante")
    # margem horizontal mais folgada (1.15) para o desenho respirar
    ax.set_xlim(-B / 2 * 1.15, B / 2 * 1.15)
    ax.set_ylim(-1.0, D * 1.04)
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.xaxis.set_minor_locator(MultipleLocator(5))
    ax.grid(True, which="major", linestyle="-", linewidth=0.5, alpha=0.4)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.4, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "08_body_plan.png"))
    plt.close(fig)


def plot_half_breadth_plan():
    """
    Half-Breadth Plan (Plano de Linhas de Agua): meia-boca (m) vs posicao
    longitudinal X (m, a partir da A.P.), uma curva para cada linha de
    agua. As curvas sao coloridas em gradiente (colormap "viridis") da
    linha de agua mais baixa (cor escura/roxa) a mais alta (cor clara/
    amarelo-esverdeada), e uma linha vertical pontilhada marca a meia-nau.

    A figura usa uma proporcao bem ALONGADA na horizontal (figsize largo
    + maior espacamento entre as marcacoes do eixo X, de 25 em 25 m), de
    modo que as linhas de agua nao fiquem comprimidas/sobrepostas e o
    afilamento de proa e popa fique claramente visivel.
    """
    from matplotlib.ticker import MultipleLocator
    fig, ax = plt.subplots(figsize=(15, 5.5))   # bem mais largo (mais espaco em X)
    x_list = np.array([STATION_X[lab] for lab in STATION_LABELS])
    cmap = plt.get_cmap("viridis")

    for k, z in enumerate(Z_GRID):
        y_list = [OFFSETS_M[lab][k] for lab in STATION_LABELS]
        cor = cmap(z / D)
        ax.plot(x_list, y_list, lw=1.6, color=cor, alpha=0.95)

    ax.axvline(X_MID, color="0.35", lw=1.1, ls="--")
    ax.text(X_MID, B / 2 * 1.02, "meia-nau", color="0.35", fontsize=9,
            ha="center", va="bottom")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=D))
    cb = fig.colorbar(sm, ax=ax, pad=0.015, fraction=0.025)
    cb.set_label("Linha de agua Z [m]")

    ax.set_xlabel("x a partir da A.P. [m]")
    ax.set_ylabel("Meia-boca [m]")
    ax.set_title("Plano de Linhas de Agua (Half-Breadth Plan)")
    # margem horizontal extra para "respirar" nas pontas
    ax.set_xlim(min(x_list) - 5, max(x_list) + 5)
    ax.set_ylim(0, B / 2 * 1.08)
    ax.xaxis.set_major_locator(MultipleLocator(25))   # marcacoes de 25 em 25 m
    ax.xaxis.set_minor_locator(MultipleLocator(5))
    ax.grid(True, which="major", linestyle="-", linewidth=0.6, alpha=0.45)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.4, alpha=0.3)
    ax.set_aspect("auto")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "09_half_breadth_plan.png"))
    plt.close(fig)


# ----------------------------------------------------------------------
# Buttock Lines (linhas longitudinais paralelas ao plano diametral, a
# uma distancia fixa "b" da linha de centro). Para cada estacao, busca-se
# a altura z em que a meia-boca da secao e exatamente igual a b (primeiro
# cruzamento ascendente a partir da quilha). O lugar geometrico desses
# pontos, ao longo do comprimento do navio, forma a buttock line.
# ----------------------------------------------------------------------
def buttock_height(row_vals_m, b):
    """
    Retorna a altura z (m) em que a meia-boca da estacao (row_vals_m,
    amostrada na grade Z_GRID) cruza o valor b (m), por interpolacao
    linear entre os pontos da grade. Caso a quilha (z=0) ja seja mais
    larga que b, o cruzamento ocorre em z=0. Caso a meia-boca nunca
    atinja b (estacao mais estreita que o buttock em toda a sua altura),
    retorna NaN (a linha fica em branco nesse trecho, como esperado).
    """
    z = np.asarray(Z_GRID, dtype=float)
    y = np.asarray(row_vals_m, dtype=float)
    if y[0] >= b:
        return 0.0
    for k in range(len(z) - 1):
        y0, y1 = y[k], y[k + 1]
        lo, hi = min(y0, y1), max(y0, y1)
        if lo <= b <= hi:
            if y1 == y0:
                return float(z[k])
            frac = (b - y0) / (y1 - y0)
            return float(z[k] + frac * (z[k + 1] - z[k]))
    return np.nan


def plot_buttock_lines():
    """
    Plano de Perfil (Sheer Plan) representado por Buttock Lines: para
    cada distancia "b" da linha de centro (5, 10, 15, 20 e 25 m), traca-se
    a curva z(x) onde o casco cruza o plano longitudinal paralelo ao
    centro a essa distancia. Esta e a representacao correta do plano de
    perfil a partir de uma tabela de offsets (estacoes x linhas de agua),
    substituindo a grade simplificada usada anteriormente.
    """
    from matplotlib.ticker import MultipleLocator
    fig, ax = plt.subplots(figsize=(15, 5.5))   # bem mais largo (mais espaco em X)
    x_list = np.array([STATION_X[lab] for lab in STATION_LABELS])
    buttocks = [5.0, 10.0, 15.0, 20.0, 25.0]
    cores = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]

    for b, cor in zip(buttocks, cores):
        z_list = np.array([buttock_height(OFFSETS_M[lab], b) for lab in STATION_LABELS])
        ax.plot(x_list, z_list, "-o", ms=3, color=cor, lw=1.8, label=f"buttock {int(b)} m")

    # linha do pontal (tampo/limite superior do casco) e meia-nau
    ax.axhline(D, color="black", lw=1.4)
    ax.axvline(X_MID, color="0.35", lw=1.1, ls="--")
    ax.text(X_MID, D * 1.03, "meia-nau", color="0.35", fontsize=9, ha="center")

    ax.set_xlabel("x a partir da A.P. [m]")
    ax.set_ylabel("Altura z [m]")
    ax.set_title("Plano de Perfil - Longitudinais (Buttock Lines)")
    ax.set_xlim(min(x_list) - 5, max(x_list) + 5)
    ax.set_ylim(0, D * 1.10)
    ax.xaxis.set_major_locator(MultipleLocator(25))
    ax.xaxis.set_minor_locator(MultipleLocator(5))
    ax.grid(True, which="major", linestyle="-", linewidth=0.6, alpha=0.45)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.4, alpha=0.3)
    ax.legend(loc="lower center", fontsize=9, ncol=5,
              bbox_to_anchor=(0.5, -0.26))
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "10_buttock_lines.png"))
    plt.close(fig)


def plot_plano_de_linhas():
    """
    PLANO DE LINHAS COMPLETO (estilo prancha de projeto), combinando em
    uma unica figura os tres planos classicos de um navio + uma caixa de
    "General Particulars":

      (1) Plano de Perfil - Longitudinais (Buttock Lines): para varias
          distancias da linha de centro, a altura z(x) onde o casco cruza
          aquele plano longitudinal (perfil lateral do casco).
      (2) Body Plan (Plano de Balizas): balizas de re a esquerda e de
          vante a direita do eixo central.
      (3) Half-Breadth Plan (Plano de Linhas de Agua): meia-boca de cada
          linha de agua ao longo do comprimento.

    DIFERENCA EM RELACAO AOS GRAFICOS INDIVIDUAIS: aqui o eixo horizontal
    dos planos longitudinais e expresso em NUMERO DE ESTACAO (station),
    e nao em metros, reproduzindo a convencao de prancha de linhas (0 =
    A.P., 10 = meia-nau, 20 = F.P.). A conversao e direta:
        numero_da_estacao = X / STATION_SPACING.
    """
    from matplotlib.ticker import MultipleLocator

    # Numero de estacao de cada coluna da tabela de offsets (eixo X dos
    # planos longitudinais). Ex.: A.P.=0, meia-nau=10, F.P.=20.
    st_num = np.array(STATION_NUMBERS, dtype=float)

    # ---- Layout: 2 linhas x 2 colunas, com larguras/alturas desiguais --
    # A coluna da direita foi ALARGADA (de 1.0 para 1.5) para o Body Plan
    # nao ficar comprimido; tambem foi aumentado o espacamento horizontal
    # (wspace) entre os subplots, dando mais "respiro" ao desenho.
    fig = plt.figure(figsize=(18, 11))
    gs = fig.add_gridspec(2, 2, width_ratios=[2.6, 1.5],
                          height_ratios=[1.0, 1.0],
                          hspace=0.30, wspace=0.26)
    ax_prof = fig.add_subplot(gs[0, 0])   # perfil / buttocks (cima-esq, largo)
    ax_body = fig.add_subplot(gs[0, 1])   # body plan (cima-dir)
    ax_half = fig.add_subplot(gs[1, 0])   # half-breadth (baixo-esq, largo)
    ax_tab = fig.add_subplot(gs[1, 1])    # tabela de particulares (baixo-dir)

    fig.suptitle("Plano de Linhas - VLCC 320.000 DWT",
                 fontsize=18, fontweight="bold", y=0.97)

    # ==================================================================
    # (1) PLANO DE PERFIL - BUTTOCK LINES (eixo X em estacoes)
    # ==================================================================
    buttocks = [2.5, 5.0, 7.5, 10.0, 12.5, 15.0, 17.5, 20.0, 22.5, 25.0, 27.5]
    cmap_b = plt.get_cmap("Blues")
    for idx, b in enumerate(buttocks):
        z_list = np.array([buttock_height(OFFSETS_M[lab], b) for lab in STATION_LABELS])
        cor = cmap_b(0.35 + 0.6 * idx / max(1, len(buttocks) - 1))
        ax_prof.plot(st_num, z_list, "-", color=cor, lw=1.1)

    ax_prof.axhline(D, color="black", lw=1.6)                 # convés (pontal D)
    ax_prof.axhline(0, color="black", lw=1.6)                 # linha de base (BL)
    ax_prof.axvline(10, color="0.4", lw=1.0, ls="--")         # meia-nau (St. 10)
    ax_prof.text(-0.2, D + 0.6, "convés (D)", fontsize=9, color="0.3")
    ax_prof.text(-0.2, 0.6, "BL", fontsize=9, color="0.3")
    ax_prof.set_title("Plano de Perfil - Longitudinais (Buttocks)", fontsize=12)
    ax_prof.set_xlabel("(Estacoes / Stations)")
    ax_prof.set_ylabel("Altura  z [m]")
    ax_prof.set_xlim(-0.6, 20.6)
    ax_prof.set_ylim(-1.5, D + 2.0)
    ax_prof.xaxis.set_major_locator(MultipleLocator(1))
    ax_prof.grid(True, linestyle=":", linewidth=0.5, alpha=0.5)

    # ==================================================================
    # (2) BODY PLAN (re a esquerda, vante a direita)
    # ==================================================================
    cor_body = "#2b4c7e"
    for lab in STATION_LABELS:
        num = STATION_NUMBERS[STATION_LABELS.index(lab)]
        y = OFFSETS_M[lab]
        y_plot = -y if num <= 10 else y     # re (<=10) a esquerda; vante a direita
        ax_body.plot(y_plot, Z_GRID, "-", color=cor_body, lw=1.0, alpha=0.9)
    ax_body.axvline(0, color="black", lw=1.3)
    ax_body.set_title("Body Plan\nre <- | -> vante", fontsize=12)
    ax_body.set_xlabel("Meia-boca [m]")
    ax_body.set_ylabel("z [m]")
    # margem horizontal mais folgada (1.18) para as balizas nao encostarem
    # nas bordas e o desenho respirar
    ax_body.set_xlim(-B / 2 * 1.18, B / 2 * 1.18)
    ax_body.set_ylim(-1.0, D * 1.04)
    ax_body.xaxis.set_major_locator(MultipleLocator(10))
    ax_body.grid(True, linestyle=":", linewidth=0.5, alpha=0.5)

    # ==================================================================
    # (3) HALF-BREADTH PLAN (eixo X em estacoes, gradiente por WL)
    # ==================================================================
    cmap_h = plt.get_cmap("viridis")
    for k, z in enumerate(Z_GRID):
        y_list = [OFFSETS_M[lab][k] for lab in STATION_LABELS]
        ax_half.plot(st_num, y_list, lw=1.3, color=cmap_h(z / D), alpha=0.95)
    ax_half.axhline(0, color="black", lw=1.6)
    ax_half.axvline(10, color="0.4", lw=1.0, ls="--")        # meia-nau
    # linhas verticais leves em cada estacao inteira (grade de balizas)
    for n in range(0, 21):
        ax_half.axvline(n, color="0.85", lw=0.5, zorder=0)
    ax_half.set_title("Plano de Linhas d'Agua (Half-Breadth Plan) - todas as WL",
                      fontsize=12)
    ax_half.set_xlabel("(Estacoes / Stations)")
    ax_half.set_ylabel("Meia-boca [m]")
    ax_half.set_xlim(-0.6, 20.6)
    ax_half.set_ylim(0, B / 2 * 1.06)
    ax_half.xaxis.set_major_locator(MultipleLocator(1))
    ax_half.grid(True, axis="y", linestyle=":", linewidth=0.5, alpha=0.5)

    # ==================================================================
    # (4) TABELA "GENERAL PARTICULARS"
    # ==================================================================
    ax_tab.axis("off")
    dados = [
        ["L.O.A (m)",  f"{LOA:.1f}"],
        ["L.B.P (m)",  f"{LBP:.1f}"],
        ["B mid (m)",  f"{B:.1f}"],
        ["D mid (m)",  f"{D:.1f}"],
        ["Td mid (m)", f"{TD:.1f}"],
        ["1 baliza (m)", f"{STATION_SPACING:.1f}"],
    ]
    tab = ax_tab.table(cellText=dados,
                       colLabels=["GENERAL PARTICULARS", ""],
                       cellLoc="left", loc="center",
                       colWidths=[0.62, 0.38])
    tab.auto_set_font_size(False)
    tab.set_fontsize(11)
    tab.scale(1.0, 1.7)
    for (r, c), cell in tab.get_celld().items():
        cell.set_edgecolor("#555555")
        if r == 0:
            cell.set_text_props(fontweight="bold")
            cell.set_facecolor("#e8e8e8")

    fig.savefig(os.path.join(OUTDIR, "12_plano_de_linhas.png"))
    plt.close(fig)


def plot_tabela_colorida(df):
    """
    Gera uma IMAGEM (PNG) da tabela hidrostatica em que o cabecalho de
    cada grandeza recebe EXATAMENTE A MESMA COR usada para a curva dessa
    grandeza no grafico geral de curvas hidrostaticas (ver CURVAS_DEF /
    CORES_CURVAS). Isso permite cruzar visualmente, com facilidade, cada
    coluna da tabela com a curva correspondente do grafico -- atendendo
    ao pedido de "deixar a tabela com as mesmas cores das curvas".

    Sao incluidas apenas as colunas que possuem curva no grafico geral
    (as de CURVAS_DEF), na mesma ordem.
    """
    colunas = [col for col, *_ in CURVAS_DEF]
    # rotulos curtos para o cabecalho (sem a notacao de escala)
    rotulos_curtos = {
        "Volume_mld": "Volume\n(m3)", "Displacement_mld": "Desloc.\n(ton)",
        "LCB": "LCB\n(m)", "LCF": "LCF\n(m)", "VCB": "VCB\n(m)",
        "KMT": "KMT\n(m)", "KML": "KML\n(m)", "MTC": "MTC\n(t.m/cm)",
        "TPC": "TPC\n(t/cm)", "Awp": "Awp\n(m2)", "WSA": "WSA\n(m2)",
        "CB": "CB", "CWP": "CWP", "CM": "CM", "CP": "CP",
    }

    sub = df[colunas].copy()
    T = df.index.values.astype(float)

    # Texto de cada celula (formatado conforme a grandeza)
    def fmt(col, v):
        if np.isnan(v):
            return "-"
        if col in ("Volume_mld", "Displacement_mld", "Awp", "WSA"):
            return f"{v:,.0f}"
        if col in ("CB", "CWP", "CM", "CP"):
            return f"{v:.4f}"
        if col in ("KML", "MTC"):
            return f"{v:.1f}"
        return f"{v:.2f}"

    cell_text = [[f"{t:.0f}"] + [fmt(col, sub.iloc[i][col]) for col in colunas]
                 for i, t in enumerate(T)]
    col_labels = ["Calado\nT (m)"] + [rotulos_curtos[c] for c in colunas]

    fig, ax = plt.subplots(figsize=(20, 9))
    ax.axis("off")
    ax.set_title("Tabela Hidrostatica - 320K VLCC\n"
                 "(cores do cabecalho = cores das curvas no grafico hidrostatico)",
                 fontsize=14, fontweight="bold", pad=18)

    tab = ax.table(cellText=cell_text, colLabels=col_labels,
                   cellLoc="center", loc="center")
    tab.auto_set_font_size(False)
    tab.set_fontsize(8.5)
    tab.scale(1.0, 1.5)

    n_linhas = len(cell_text)
    # Cores do cabecalho: 1a coluna (calado) em cinza; demais com a cor
    # da respectiva curva (texto branco para contraste).
    for j in range(len(col_labels)):
        cell = tab[0, j]
        if j == 0:
            cell.set_facecolor("#444444")
        else:
            cell.set_facecolor(CORES_CURVAS[j - 1])
        cell.set_text_props(color="white", fontweight="bold")
        cell.set_height(0.09)

    # Zebra striping (linhas alternadas) no corpo, para leitura mais facil
    for i in range(1, n_linhas + 1):
        for j in range(len(col_labels)):
            cell = tab[i, j]
            cell.set_facecolor("#f4f4f4" if i % 2 == 0 else "white")
            cell.set_edgecolor("#cccccc")
            if j == 0:
                cell.set_text_props(fontweight="bold")

    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "11_tabela_hidrostatica_colorida.png"))
    plt.close(fig)


def gerar_todos_os_graficos(df):
    print()
    print("Gerando graficos (PNG)...")
    plot_curva_geral(df)
    plot_simples(df, "Volume_mld", "Volume Moldado x Calado", "Volume (m^3)",
                 "02_volume_x_calado.png")
    plot_simples(df, "Displacement_mld", "Deslocamento Moldado x Calado",
                 "Deslocamento (ton)", "03_deslocamento_x_calado.png")
    plot_simples(df, "KMT", "KMT x Calado", "KMT (m)", "04_kmt_x_calado.png")
    plot_simples(df, "KML", "KML x Calado", "KML (m)", "05_kml_x_calado.png")
    plot_simples(df, "MTC", "MTC x Calado", "MTC (ton.m/cm)", "06_mtc_x_calado.png")
    plot_simples(df, "TPC", "TPC x Calado", "TPC (ton/cm)", "07_tpc_x_calado.png")
    plot_body_plan()
    plot_half_breadth_plan()
    plot_buttock_lines()
    plot_plano_de_linhas()
    plot_tabela_colorida(df)
    print(f"Graficos salvos na pasta: ./{OUTDIR}/")


# ======================================================================
# 9. EXPORTACAO DA TABELA HIDROSTATICA (CSV e XLSX)
# ======================================================================
def exportar_tabela(df):
    caminho_csv = os.path.join(OUTDIR, "tabela_hidrostatica.csv")
    df.to_csv(caminho_csv, float_format="%.4f")
    print(f"\nTabela hidrostatica exportada em: {caminho_csv}")

    caminho_xlsx = os.path.join(OUTDIR, "tabela_hidrostatica.xlsx")
    try:
        df.to_excel(caminho_xlsx, sheet_name="Hidrostatica")
        print(f"Tabela hidrostatica exportada em: {caminho_xlsx}")
    except Exception as e:
        print(f"[AVISO] Nao foi possivel gerar o arquivo .xlsx ({e}). "
              f"O arquivo .csv foi gerado normalmente.")


# ======================================================================
# 10. PROGRAMA PRINCIPAL
# ======================================================================
def main():
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 30)

    # Passo 1-7 (procedimento do PDF): construir/integrar a tabela hidrostatica
    df = montar_tabela_hidrostatica()

    print()
    print("=" * 78)
    print("TABELA HIDROSTATICA CALCULADA (resumo)")
    print("=" * 78)
    colunas_resumo = ["Volume_mld", "Volume_ext", "Displacement_mld", "Displacement_ext",
                       "LCB", "LCF", "VCB", "KMT", "KML", "MTC", "TPC", "WSA",
                       "CB", "CWP", "CM", "CP"]
    print(df[colunas_resumo].round(3).to_string())

    # Exportacao
    exportar_tabela(df)

    # Graficos
    gerar_todos_os_graficos(df)

    # Validacao com a tabela de referencia do PDF
    validar_resultados(df)

    print()
    print("=" * 78)
    print("PROCESSAMENTO CONCLUIDO.")
    print(f"Todos os arquivos de saida estao na pasta: ./{OUTDIR}/")
    print("=" * 78)


if __name__ == "__main__":
    main()