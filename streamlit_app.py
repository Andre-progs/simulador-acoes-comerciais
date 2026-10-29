"""
Simulador de Ações Comerciais (versão pública)
Todos os dados são fictícios: produtos, clientes, preços e volumes são gerados pelo próprio app.
Rodar: python -m streamlit run "Simulador de ação comercial.py"
"""
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st

MESES = ["jan", "fev", "mar", "abr", "mai", "jun", "jul", "ago", "set", "out", "nov", "dez"]
N_SIM = 2000

# ================================================================ base fictícia
# nome, canais, elasticidade real, margem, tendência anual, sazonalidade, mês de pico, volume/mês, pré-compra, SKUs
FAMILIAS = [
    ("Solenne", ("FARMÁCIA", "PERFUMARIA", "E-COMMERCE"), -1.6, 0.62, 0.08, 0.45, 1, 40000, 0.35,
     [("Solenne FPS 50 fluido toque seco 50 ml", 79.90, 0.6), ("Solenne FPS 70 com cor 40 ml", 89.90, 0.4)]),
    ("Hidrava", ("FARMÁCIA", "PERFUMARIA"), -1.3, 0.58, 0.03, 0.25, 7, 30000, 0.30,
     [("Hidrava loção corporal 400 ml", 59.90, 0.65), ("Hidrava creme para mãos 50 g", 29.90, 0.35)]),
    ("Clarivé", ("FARMÁCIA", "PERFUMARIA", "E-COMMERCE"), -1.0, 0.70, 0.12, 0.10, 5, 12000, 0.25,
     [("Clarivé sérum vitamina C 30 ml", 149.90, 0.55), ("Clarivé retinol noturno 30 ml", 169.90, 0.45)]),
    ("Acnelys", ("FARMÁCIA", "E-COMMERCE"), -1.2, 0.60, 0.05, 0.08, 2, 20000, 0.25,
     [("Acnelys gel de limpeza 150 g", 54.90, 0.6), ("Acnelys secativo 15 g", 39.90, 0.4)]),
    ("Velura", ("FARMÁCIA", "E-COMMERCE"), -0.9, 0.64, 0.06, 0.05, 9, 14000, 0.20,
     [("Velura shampoo antiqueda 200 ml", 69.90, 0.6), ("Velura tônico capilar 100 ml", 119.90, 0.4)]),
    ("Dermaflor", ("FARMÁCIA",), -0.7, 0.66, 0.04, 0.06, 6, 9000, 0.15,
     [("Dermaflor bálsamo reparador 40 g", 64.90, 1.0)]),
    ("Nutrilume", ("FARMÁCIA", "E-COMMERCE"), -1.5, 0.55, 0.15, 0.06, 1, 10000, 0.30,
     [("Nutrilume colágeno + ácido hialurônico 30 sachês", 129.90, 1.0)]),
]
CLIENTES = {
    "FARMÁCIA": ["Rede Farmavale", "Drogarias Sol Nascente", "Rede Bem Cuidar", "Farmácias Primavera",
                 "Drogaria Ponte Alta", "Rede Viva Saúde"],
    "PERFUMARIA": ["Perfumarias Lírio Branco", "Casa Aroma Perfumaria", "Perfumaria Brisa do Mar"],
    "E-COMMERCE": ["Loja Online Dermaclick", "Marketplace Vitrine Beleza"],
}


@st.cache_data(show_spinner=False)
def gerar_base(seed=42):
    rng = np.random.default_rng(seed)
    meses = pd.date_range("2023-01-01", "2026-09-01", freq="MS")
    T = len(meses)
    linhas = []
    for fam, canais, eps, marg, tend, amp, pico, vol, pre, skus in FAMILIAS:
        clientes = [c for cn in canais for c in CLIENTES[cn]]
        canal_de = {c: cn for cn in canais for c in CLIENTES[cn]}
        tam = rng.dirichlet(np.ones(len(clientes)) * 2.5)
        desc = np.zeros((len(clientes), T))
        bonif = np.zeros((len(clientes), T))
        t = int(rng.integers(2, 6))
        while t < T:
            dur = int(rng.integers(1, 3))
            alvo = rng.random(len(clientes)) < 0.55
            desc[alvo, t:t + dur] = rng.choice([0.05, 0.08, 0.10, 0.12, 0.15, 0.20])
            bonif[alvo, t:t + dur] = rng.choice([0.0, 0.0, 0.0, 0.05, 0.10])
            t += dur + int(rng.integers(3, 8))
        for sku, preco, peso in skus:
            custo = preco * (1 - marg)
            for ci, cli in enumerate(clientes):
                base = vol * peso * tam[ci]
                estoque = base * 2.5
                for k, m in enumerate(meses):
                    reajustes = (m.year - 2023) + (1 if m.month >= 4 else 0)
                    lista = preco * 1.045 ** reajustes
                    saz = 1 + amp * np.cos(2 * np.pi * (m.month - pico) / 12)
                    nivel = base * (1 + tend) ** (k / 12) * saz
                    d, b = desc[ci, k], bonif[ci, k]
                    d_ef = 1 - (1 - d) / (1 + b)
                    so = nivel * (1 - d_ef) ** eps * rng.lognormal(0, 0.10)
                    si = so + 0.5 * (2.5 * nivel - estoque)
                    if d_ef > 0:
                        si += pre * nivel
                    si = max(0.0, si * rng.lognormal(0, 0.20))
                    bon_un = b * si
                    estoque = max(0.0, estoque + si + bon_un - so)
                    linhas.append((int(m.strftime("%Y%m")), m, cli, canal_de[cli], sku, fam, round(si), round(bon_un),
                                   round(so), round(estoque), round(lista, 2), d, b, round(custo, 2)))
    return pd.DataFrame(linhas, columns=["PERIODO", "mes", "Cliente", "Canal", "SKU", "FAMILIA", "Sell_In",
                                         "Bonificado", "Sell_Out", "Estoque", "Preco_Lista", "Desconto",
                                         "Bonif_Pct", "Custo"])


# ================================================================ o que a base ensina
def preco_efetivo(d):
    return (1 - d["Desconto"]) / (1 + d["Bonif_Pct"])


@st.cache_data(show_spinner=False)
def aprender(base):
    """Mede, por produto, quanto o volume reagiu aos descontos e bonificações do passado."""
    out = {}
    for fam, d in base.groupby("FAMILIA"):
        d = d[d.Sell_Out > 0].copy()
        x = np.log(preco_efetivo(d))
        y = np.log(d.Sell_Out)
        g = d.Cliente + "|" + d.SKU
        # compara o mesmo cliente com e sem ação, descontando o que aconteceu no mercado naquele mês
        xd, yd = x.copy(), y.copy()
        for _ in range(15):
            xd = xd - xd.groupby(g).transform("mean")
            xd = xd - xd.groupby(d.PERIODO).transform("mean")
            yd = yd - yd.groupby(g).transform("mean")
            yd = yd - yd.groupby(d.PERIODO).transform("mean")
        sxx = float((xd ** 2).sum())
        acoes = int((d.Desconto + d.Bonif_Pct > 0).sum())
        if sxx > 1e-6 and acoes >= 5:
            b = float((xd * yd).sum() / sxx)
            se = float(np.sqrt(((yd - b * xd) ** 2).sum() / (len(d) - 2) / sxx))
        else:
            b, se = -0.8, 10.0
        # referência neutra (-0,8 ± 0,8) só pesa quando a base tem pouca evidência
        p0, p1 = 1 / 0.8 ** 2, 1 / se ** 2
        el_mu = (-0.8 * p0 + b * p1) / (p0 + p1)
        el_sd = np.sqrt(1 / (p0 + p1))

        # pré-compra: quanto o sell-in passa do sell-out em meses de ação
        ref = d[d.Desconto + d.Bonif_Pct == 0].groupby(g).Sell_Out.mean()
        d["ref"] = g.map(ref)
        d["gap"] = (d.Sell_In - d.Sell_Out) / d["ref"]
        em_acao = d.Desconto + d.Bonif_Pct > 0
        pre = float(np.clip(d.loc[em_acao, "gap"].mean() - d.loc[~em_acao, "gap"].mean(), 0, 0.8)) if em_acao.any() else 0.15
        out[fam] = dict(el_mu=float(el_mu), el_sd=float(el_sd), n_acoes=acoes, pre=pre * 100)
    return out


# ================================================================ linha de base
MODELOS = [("nivel", 12), ("tendencia", 24), ("sazonal", 36)]
ROTULO = {"nivel": "nível recente", "tendencia": "tendência", "sazonal": "tendência com sazonalidade"}


def br(x, dec=0):
    return f"{x:,.{dec}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def pc(x):
    return f"{x:+.1f}%".replace(".", ",")


def rot(ts):
    return f"{MESES[ts.month - 1]}/{ts.year}"


def serie_mensal(d, eps):
    """Soma o recorte por mês e tira o efeito das ações passadas, para a linha de base não herdar picos de promoção."""
    pe = preco_efetivo(d)
    d = d.assign(so_limpo=d.Sell_Out / pe ** eps, so_acao=np.where(pe < 1, d.Sell_Out, 0))
    g = d.groupby("mes").agg(so=("so_limpo", "sum"), so_real=("Sell_Out", "sum"), si=("Sell_In", "sum"),
                             est=("Estoque", "sum"), acao=("so_acao", "sum"))
    return g


def indice_t(idx):
    return (idx.year * 12 + idx.month - 1).values.astype(int)


def desenho(tipo, t, ref):
    X = [np.ones(len(t))]
    if tipo in ("tendencia", "sazonal"):
        X.append((t - ref) / 12.0)
    X = np.column_stack(X)
    if tipo == "sazonal":
        S = np.zeros((len(t), 12))
        S[np.arange(len(t)), t % 12] = 1.0
        X = np.column_stack([X, S - 1 / 12])
    return X


def ajustar(y, t, tipo, janela):
    y, t = y[-janela:], t[-janela:]
    ly = np.log(np.maximum(y, 1.0))
    ref = t[-1]
    X = desenho(tipo, t, ref)
    R = np.zeros((X.shape[1], X.shape[1]))
    if tipo == "sazonal":
        R[2:, 2:] = np.eye(12) * 2.0
    A = np.linalg.inv(X.T @ X + R)
    beta = A @ X.T @ ly
    res = ly - X @ beta
    gl = max(len(ly) - np.trace(X @ A @ X.T), 3.0)
    sigma = float(np.sqrt((res ** 2).sum() / gl))
    return dict(tipo=tipo, beta=beta, A=A, sigma=sigma, res=res, ref=ref)


def prever(m, th, n, rng):
    Xf = desenho(m["tipo"], th, m["ref"])
    k = Xf.shape[1]
    cov = (m["sigma"] ** 2) * m["A"]
    cov = (cov + cov.T) / 2 + 1e-10 * np.eye(k)
    B = m["beta"] + rng.standard_normal((n, k)) @ np.linalg.cholesky(cov).T
    res = m["res"] * (m["sigma"] / max(m["res"].std(), 1e-9))
    return np.exp(B @ Xf.T + rng.choice(res, size=(n, len(th))))


def backtest(y, t, h=6):
    """Testa os métodos nos últimos 6 meses (que eles não viram) e fica com o que errou menos."""
    if len(y) < 24:
        return "nivel", None
    real = y[-h:]
    melhor = None
    for tipo, j in MODELOS:
        m = ajustar(y[:-h], t[:-h], tipo, j)
        pred = np.exp(desenho(tipo, t[-h:], m["ref"]) @ m["beta"])
        mape = float(np.mean(np.abs(pred - real) / np.maximum(real, 1.0)))
        if melhor is None or mape < melhor[1]:
            melhor = (tipo, mape)
    return melhor


# ================================================================ simulação
def simular(m, ultimo_t, s, H, desc, bonif, inicio, dur, el_mu, el_sd, pre, pb, P0, margem, seed=7):
    rng = np.random.default_rng(seed)
    n = N_SIM
    th = np.arange(ultimo_t + 1, ultimo_t + 1 + H)
    so = prever(m, th, n, rng)

    recente = s.tail(12)
    rho = float(np.clip(recente.si.sum() / max(recente.so.sum(), 1.0), 0.3, 2.0))
    razao = np.log(np.maximum(recente.si.values, 1) / np.maximum(rho * recente.so.values, 1))
    sig = float(np.clip(razao.std(), 0.05, 0.8))
    si = rho * so * np.exp(rng.normal(0, sig, (n, H)) - sig ** 2 / 2)

    d, b = desc / 100.0, bonif / 100.0
    d_ef = 1 - (1 - d) / (1 + b)
    mask = np.zeros(H)
    mask[inicio:inicio + dur] = 1.0
    eps = np.clip(rng.normal(el_mu, el_sd, n), -6, 0)
    lift = ((1 - d_ef) ** eps[:, None] - 1) * mask[None, :]

    so_c = so * (1 + lift)
    extra = (pre / 100.0) * (si / rho) * mask[None, :] if d_ef > 0 else np.zeros_like(si)
    antecipado = extra.sum(1)
    devolve = np.zeros_like(si)
    fim = min(inicio + dur, H)
    for j in range(fim, min(fim + pb, H)):
        devolve[:, j] = antecipado / pb
    si_c = np.maximum(si * (1 + lift) + extra - devolve, 0)

    c = (1 - margem / 100.0) * P0
    preco = P0 * (1 - d * mask)
    rec0, rec1 = si * P0, si_c * preco[None, :]
    lb0 = si * (P0 - c)
    lb1 = si_c * (preco[None, :] - c) - b * si_c * mask[None, :] * c
    dias_extra = np.cumsum(extra - devolve, axis=1) / (so.mean(1, keepdims=True) / 30.0)
    return dict(so=so, so_c=so_c, si=si, si_c=si_c, rec0=rec0, rec1=rec1, lb0=lb0, lb1=lb1,
                dias_extra=dias_extra, c=c, rho=rho)


def resumo(a, b):
    ta, tb = a.sum(1), b.sum(1)
    delta = tb - ta
    pct = delta / np.maximum(ta, 1e-9) * 100
    return dict(cen=float(np.median(tb)), p10=float(np.percentile(pct, 10)), p50=float(np.median(pct)),
                p90=float(np.percentile(pct, 90)), prob=float(np.mean(delta > 0)))


# ================================================================ interface
st.set_page_config(page_title="Simulador de Ações Comerciais", layout="wide")
st.title("Simulador de Ações Comerciais")

base = gerar_base()
params = aprender(base)
ult_mes = base["mes"].max()
st.caption(f"Dados fictícios para demonstração. Histórico de {rot(base['mes'].min())} a {rot(ult_mes)}, "
           f"{base['SKU'].nunique()} apresentações, {base['Cliente'].nunique()} clientes.")

st.subheader("Escopo")
vol = (base.Sell_Out * base.Preco_Lista).groupby(base.FAMILIA).sum().sort_values(ascending=False)
c1, c2, c3, c4 = st.columns(4)
fam = c1.selectbox("Produto", vol.index.tolist())
d_f = base[base.FAMILIA == fam]
sku = c2.selectbox("Apresentação", ["Todas"] + sorted(d_f.SKU.unique()))
d_s = d_f if sku == "Todas" else d_f[d_f.SKU == sku]
canal = c3.selectbox("Canal", ["Todos"] + sorted(d_s.Canal.unique()))
d_c = d_s if canal == "Todos" else d_s[d_s.Canal == canal]
cli = c4.selectbox("Cliente", ["Todos"] + sorted(d_c.Cliente.unique()))
d_x = d_c if cli == "Todos" else d_c[d_c.Cliente == cli]

P = params[fam]
s = serie_mensal(d_x, P["el_mu"])
y, t_idx = s.so.values, indice_t(s.index)
tipo, mape = backtest(y, t_idx)
modelo = ajustar(y, t_idx, tipo, dict(MODELOS)[tipo])

ult = d_x[d_x.mes == ult_mes]
peso = ult.groupby("SKU").Sell_In.sum().clip(lower=1)
P0 = float((ult.groupby("SKU").Preco_Lista.mean() * peso).sum() / peso.sum())
custo = float((ult.groupby("SKU").Custo.mean() * peso).sum() / peso.sum())
margem = (1 - custo / P0) * 100
cobertura_hoje = float(s.est.iloc[-1] / max(s.so_real.tail(3).mean(), 1) * 30)

st.subheader("Ação")
rotulos = [rot(ult_mes + pd.DateOffset(months=i + 1)) for i in range(12)]
a1, a2, a3, a4, a5 = st.columns(5)
desc = a1.slider("Desconto (%)", 0.0, 40.0, 10.0, 0.5, format="%.1f")
bonif = a2.slider("Bonificação (%)", 0.0, 30.0, 0.0, 0.5, format="%.1f",
                  help="Unidades grátis para cada 100 unidades compradas.")
H = a5.selectbox("Horizonte (meses)", [3, 6, 9, 12], index=1)
inicio = a3.selectbox("Início da ação", rotulos[:H])
dur = a4.slider("Duração (meses)", 1, 6, 2)
inicio_i = rotulos.index(inicio)
dur = min(dur, H - inicio_i)

r = simular(modelo, int(t_idx[-1]), s, H, desc, bonif, inicio_i, dur, P["el_mu"], P["el_sd"], P["pre"], 3, P0, margem)
meses_fc = pd.date_range(ult_mes + pd.DateOffset(months=1), periods=H, freq="MS")
R_so, R_si = resumo(r["so"], r["so_c"]), resumo(r["si"], r["si_c"])
R_rec, R_lb = resumo(r["rec0"], r["rec1"]), resumo(r["lb0"], r["lb1"])

st.subheader(f"Resultado em {H} meses")


def cartao(col, titulo, R, moeda=False):
    f = (lambda v: "R$ " + br(v)) if moeda else br
    col.metric(titulo, f(R["cen"]), f"{pc(R['p50'])} vs sem ação")
    col.caption(f"Faixa provável: {pc(R['p10'])} a {pc(R['p90'])}")


k1, k2, k3, k4 = st.columns(4)
cartao(k1, "Sell-out (unid.)", R_so)
cartao(k2, "Sell-in (unid.)", R_si)
cartao(k3, "Receita", R_rec, True)
cartao(k4, "Lucro bruto", R_lb, True)

if desc == 0 and bonif == 0:
    st.info("Defina um desconto ou uma bonificação para ver o impacto.")
else:
    d, b = desc / 100, bonif / 100
    den = P0 * (1 - d) - r["c"] * (1 + b)
    fim_acao = rotulos[inicio_i + dur - 1]
    periodo = f"em {inicio}" if dur == 1 else f"de {inicio} a {fim_acao}"
    txt = ("Com " + (f"{br(desc, 1)}% de desconto" if desc else "")
           + (" e " if desc and bonif else "") + (f"{br(bonif, 1)}% de bonificação" if bonif else "")
           + f" {periodo}, o sell-out acumulado deve variar entre {pc(R_so['p10'])} e {pc(R_so['p90'])} "
           f"(mediana {pc(R_so['p50'])}). A chance de o lucro bruto superar o cenário sem ação é de "
           f"{R_lb['prob'] * 100:.0f}%.")
    if den > 0:
        txt += f" Para empatar o lucro, o volume precisa subir pelo menos {br(((P0 - r['c']) / den - 1) * 100, 1)}% durante a ação."
    else:
        txt += " Nessa combinação, cada unidade vendida dá prejuízo bruto: nenhum ganho de volume compensa."
    txt += (f" Os clientes têm hoje cerca de {br(cobertura_hoje)} dias de estoque e a pré-compra "
            f"pode somar mais {br(float(np.median(r['dias_extra'].max(1))))} dias no pico.")
    st.markdown(txt)


def grafico(hist, acoes, base_p, cen_p, titulo):
    h = hist.tail(24).rename("valor").rename_axis("mes").reset_index()
    fc = pd.DataFrame({"mes": meses_fc, "b10": base_p[0], "b50": base_p[1], "b90": base_p[2],
                       "c10": cen_p[0], "c50": cen_p[1], "c90": cen_p[2]})
    v = h["valor"].iloc[-1]
    fc = pd.concat([pd.DataFrame({"mes": [h["mes"].iloc[-1]], **{c: [v] for c in fc.columns[1:]}}), fc],
                   ignore_index=True)
    marc = pd.DataFrame({"mes": acoes[acoes.index >= h["mes"].iloc[0]].index})
    x = alt.X("mes:T", title=None, axis=alt.Axis(format="%m/%y"))
    yq = lambda c: alt.Y(f"{c}:Q", title=None)
    cb, cc = "#6b7a90", "#e8590c"
    camadas = [
        alt.Chart(marc).mark_rule(color=cc, opacity=0.18, strokeWidth=8).encode(x=x),
        alt.Chart(fc).mark_area(opacity=0.16, color=cb).encode(x=x, y=yq("b10"), y2="b90:Q"),
        alt.Chart(fc).mark_area(opacity=0.22, color=cc).encode(x=x, y=yq("c10"), y2="c90:Q"),
        alt.Chart(h).mark_line(color="#334155").encode(x=x, y=yq("valor")),
        alt.Chart(fc).mark_line(color=cb, strokeDash=[5, 4]).encode(x=x, y=yq("b50")),
        alt.Chart(fc).mark_line(color=cc, strokeWidth=2.5).encode(x=x, y=yq("c50")),
    ]
    return alt.layer(*camadas).properties(height=270, title=titulo)


pct = lambda a: np.percentile(a, [10, 50, 90], axis=0)
acoes = s.acao[s.acao / s.so_real.clip(lower=1) > 0.3]
g1, g2 = st.columns(2)
g1.altair_chart(grafico(s.so_real, acoes, pct(r["so"]), pct(r["so_c"]), "Sell-out mensal (unid.)"), width="stretch")
g2.altair_chart(grafico(s.si, acoes, pct(r["si"]), pct(r["si_c"]), "Sell-in mensal (unid.)"), width="stretch")
st.caption("Linha tracejada: sem ação. Linha laranja: com a ação. Faixas: 80% das simulações. "
           "Barras laranja claras no histórico: meses com ações passadas.")

with st.expander("Detalhe mensal"):
    bo, co, bi, ci = pct(r["so"]), pct(r["so_c"]), pct(r["si"]), pct(r["si_c"])
    st.dataframe(pd.DataFrame({
        "Mês": [rot(m) for m in meses_fc],
        "Sell-out sem ação": bo[1].round(0), "Sell-out com ação": co[1].round(0),
        "Sell-in sem ação": bi[1].round(0), "Sell-in com ação": ci[1].round(0),
        "Estoque extra (dias)": np.median(r["dias_extra"], axis=0).round(1)}), hide_index=True, width="stretch")

with st.expander("Como o simulador calcula"):
    erro = f"{mape * 100:.0f}%" if mape is not None else "n/d"
    st.markdown(f"""
- **O que a base ensinou sobre {fam}**: em {P['n_acoes']} meses-cliente com ação, cada 1% de desconto efetivo moveu o volume em
  {br(-P['el_mu'], 2)}% (incerteza ±{br(P['el_sd'], 2)}). A conta compara o mesmo cliente com e sem ação no mesmo mês de mercado.
  Pré-compra observada: {br(P['pre'], 0)}% do volume mensal. Margem bruta: {br(margem, 0)}%.
- **Linha de base**: histórico sem o efeito das promoções passadas, projetado pelo método que errou menos nos últimos 6 meses
  ({ROTULO[tipo]}, erro de {erro}).
- **Efeito da ação**: volume × (1 − desconto efetivo) ^ sensibilidade. Bonificação vira desconto efetivo: 1 − (1 − desconto) ÷ (1 + bonificação).
- **Sell-in**: acompanha o sell-out, soma a pré-compra durante a ação e devolve esse volume nos 3 meses seguintes.
- **Incerteza**: {br(N_SIM)} simulações combinando o erro da linha de base e a dúvida sobre a sensibilidade. As faixas vão de P10 a P90.
""")
