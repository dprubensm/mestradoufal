import os
import glob
import argparse
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import ttest_rel, wilcoxon, shapiro
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

DEFAULT_INPUT  = r"C:\Users\RUBENS\PycharmProjects\EMGAnalise\Dados_tratados_input"
DEFAULT_OUTPUT = r"C:\Users\RUBENS\PycharmProjects\EMGAnalise\Dados_tratados_output"

def sanitize(name):
    return re.sub(r'[^0-9A-Za-z_]', '_', name)

def clean_columns(df):
    # padroniza colunas: strip, lower, não-alfanum -> _
    df = df.rename(columns=lambda c: re.sub(r'[^0-9a-z]', '_', str(c).strip().lower()))
    return df

def load_data(input_folder, sheet80='Teste 80%', sheet120='Teste 120%'):
    files = glob.glob(os.path.join(input_folder, "*.xlsx"))
    dfs80, dfs120, ids = [], [], []
    for f in files:
        ids.append(os.path.splitext(os.path.basename(f))[0])
        xls = pd.ExcelFile(f)
        df80 = pd.read_excel(xls, sheet80)
        df80 = clean_columns(df80)
        dfs80.append(df80)
        df120 = pd.read_excel(xls, sheet120)
        df120 = clean_columns(df120)
        dfs120.append(df120)
    return ids, dfs80, dfs120

def compute_slope(df, var):
    if var not in df.columns or 'rel_time_sec' not in df.columns:
        return np.nan
    sub = df.dropna(subset=[var, 'rel_time_sec'])
    if len(sub) > 2:
        m, _ = np.polyfit(sub['rel_time_sec'], sub[var], 1)
        return m
    return np.nan

def trend_analysis(ids, dfs80, dfs120, variables):
    rows = []
    for var in variables:
        slopes80 = [compute_slope(df, var) for df in dfs80]
        slopes120= [compute_slope(df, var) for df in dfs120]
        paired = [(a, b) for a, b in zip(slopes80, slopes120)
                  if not np.isnan(a) and not np.isnan(b)]
        if len(paired) < 3:
            continue
        a, b = zip(*paired)
        diff = np.array(a) - np.array(b)
        p_sh = shapiro(diff)[1]
        if p_sh >= 0.05:
            stat, p = ttest_rel(a, b); test = 't‑test pareado'
        else:
            stat, p = wilcoxon(a, b); test = 'Wilcoxon pareado'
        rows.append({
            'variavel': var,
            'slope80': np.mean(a),
            'slope120': np.mean(b),
            'teste': test,
            'estat': stat,
            'p_value': p
        })
    return pd.DataFrame(rows)

def segment_correlation(df, var):
    if var not in df.columns:
        return np.nan
    rhos = []
    for seg, sub in df.groupby('segment'):
        if len(sub) > 3:
            rhos.append(sub[var].corr(sub['emg_rms']))
    return np.nanmean(rhos) if rhos else np.nan

def corr_analysis(dfs80, dfs120, variables):
    rows = []
    for var in variables:
        r80 = [segment_correlation(df, var) for df in dfs80]
        r120= [segment_correlation(df, var) for df in dfs120]
        paired = [(a, b) for a, b in zip(r80, r120)
                  if not np.isnan(a) and not np.isnan(b)]
        if len(paired) < 3:
            continue
        a, b = zip(*paired)
        za = np.arctanh(a); zb = np.arctanh(b)
        stat, p = ttest_rel(za, zb); test = 't‑test z'
        rows.append({
            'variavel': var,
            'r80_mean': np.mean(a),
            'r120_mean': np.mean(b),
            'teste': test,
            'estat': stat,
            'p_value': p
        })
    return pd.DataFrame(rows)

def mixed_effects_analysis(ids, dfs80, dfs120, variables):
    df_all = []
    for id_, df in zip(ids, dfs80):
        tmp = df.copy(); tmp['cond'] = '80%'; tmp['indiv'] = id_
        df_all.append(tmp)
    for id_, df in zip(ids, dfs120):
        tmp = df.copy(); tmp['cond'] = '120%'; tmp['indiv'] = id_
        df_all.append(tmp)
    df_concat = pd.concat(df_all, ignore_index=True)

    results = []
    for var in variables:
        if var not in df_concat.columns:
            continue
        sub = df_concat.dropna(subset=[var, 'emg_rms'])
        if len(sub) < 10:
            continue
        m = smf.mixedlm(f"emg_rms ~ {var}*cond", sub, groups=sub["indiv"]).fit()
        results.append({
            'variavel': var,
            'coef_inter': m.params.get(f"{var}:cond[T.120%]", np.nan),
            'p_inter': m.pvalues.get(f"{var}:cond[T.120%]", np.nan),
            'model': m
        })
    return results

def plot_scatter_group(df_concat, var, out_folder):
    os.makedirs(out_folder, exist_ok=True)
    x = df_concat.get(var, pd.Series(dtype=float))
    y = df_concat['emg_rms']
    plt.figure(figsize=(6,4))
    plt.scatter(x, y, alpha=0.5)
    sub = df_concat.dropna(subset=[var, 'emg_rms'])
    if len(sub) > 2:
        m, b = np.polyfit(sub[var], sub['emg_rms'], 1)
        plt.plot(sub[var], m*sub[var] + b, 'r-')
    plt.title(var)
    fn = os.path.join(out_folder, f"{sanitize(var)}.png")
    plt.tight_layout(); plt.savefig(fn); plt.close()
    return fn

def write_report(writer, df_trend, df_corr, df_mix, plot_files):
    df_trend.to_excel(writer, sheet_name='Tendencias', index=False)
    df_corr.to_excel(writer, sheet_name='Correlacoes', index=False)
    df_me = pd.DataFrame([{
        'variavel': r['variavel'],
        'coef_inter': r['coef_inter'],
        'p_inter': r['p_inter']
    } for r in df_mix])
    df_me.to_excel(writer, sheet_name='Mixed_Effects', index=False)

    ws = writer.book.add_worksheet('Graficos')
    writer.sheets['Graficos'] = ws
    row = 0
    for var, img in plot_files.items():
        ws.insert_image(row, 0, img, {'x_scale': 0.5})
        row += 20

    ws2 = writer.book.add_worksheet('Resumo_Texto')
    writer.sheets['Resumo_Texto'] = ws2
    row = 0

    sig = []
    for _, x in df_trend.iterrows():
        if x.p_value < 0.05:
            sig.append(f"{x.variavel} (tendência, p={x.p_value:.3f})")
    for _, x in df_corr.iterrows():
        if x.p_value < 0.05:
            sig.append(f"{x.variavel} (correlação, p={x.p_value:.3f})")
    for m in df_mix:
        if m['p_inter'] < 0.05:
            sig.append(f"{m['variavel']} (interação mista, p={m['p_inter']:.3f})")

    if sig:
        texto = ("Foram encontradas relações estatisticamente significativas em: "
                 + "; ".join(sig) + ".")
    else:
        texto = "Nenhuma relação estatisticamente significativa foi identificada."
    ws2.write(row, 0, texto); row += 2

    if sig:
        conclusao = (
            "Conclusão: Estes resultados indicam que certas variáveis respiratórias "
            "mostram alterações de tendência, correlação e/ou interações com a condição "
            "de esforço, sugerindo mecanismos de adaptação ou desacoplamento entre ventilação "
            "e atividade muscular. Estes achados podem orientar intervenções ou estudos futuros."
        )
    else:
        conclusao = (
            "Conclusão: Não foram observadas relações estatisticamente robustas entre as "
            "variáveis respiratórias e a atividade muscular (EMG_RMS) nas condições avaliadas."
        )
    ws2.write(row, 0, conclusao)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", default=DEFAULT_INPUT)
    parser.add_argument("--out_folder", default=DEFAULT_OUTPUT)
    parser.add_argument("--sheet80", default="Teste 80%")
    parser.add_argument("--sheet120", default="Teste 120%")
    args = parser.parse_args()

    os.makedirs(args.out_folder, exist_ok=True)
    ids, dfs80, dfs120 = load_data(args.input_folder, args.sheet80, args.sheet120)

    exclude = {'t','time_sec','rel_time_sec','emg_rms','segment'}
    vars_all = set().union(*(set(df.columns) for df in dfs80))
    variables = []
    for v in vars_all:
        if v in exclude:
            continue
        for df in dfs80:
            if v in df.columns and np.issubdtype(df[v].dtype, np.number):
                variables.append(v)
                break
    variables = sorted(variables)

    df_trend = trend_analysis(ids, dfs80, dfs120, variables)
    df_corr  = corr_analysis(dfs80, dfs120, variables)
    df_mix   = mixed_effects_analysis(ids, dfs80, dfs120, variables)

    df_concat = pd.concat(dfs80 + dfs120, ignore_index=True)
    plot_folder = os.path.join(args.out_folder, 'plots')
    plot_files = {v: plot_scatter_group(df_concat, v, plot_folder)
                  for v in variables}

    out = os.path.join(args.out_folder, 'analise_avancada_grupo.xlsx')
    with pd.ExcelWriter(out, engine='xlsxwriter') as writer:
        write_report(writer, df_trend, df_corr, df_mix, plot_files)

    print("Análise avançada de grupo salva em:", out)

if __name__ == "__main__":
    main()
