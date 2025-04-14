import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from scipy.stats import ttest_rel
import xlsxwriter

# Suprime warnings indesejados (por exemplo, dos interpolações)
warnings.filterwarnings("ignore", category=RuntimeWarning)


# ---------- Função auxiliar para sanitizar nomes de arquivo ----------
def sanitize_filename(filename):
    forbidden = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
    for char in forbidden:
        filename = filename.replace(char, '_')
    return filename


# ---------- Função auxiliar: Converter índice para letra (para gráficos no Excel) ----------
def col_to_excel(col_index):
    letters = ""
    while col_index >= 0:
        letters = chr(col_index % 26 + 65) + letters
        col_index = col_index // 26 - 1
    return letters


# ---------- Sincronização e Criação do Sinal EMG ----------
def create_step_emg(time_array, total_time, emg_segments, n_segments=10):
    """
    Cria uma step function para os dados de EMG.

    Cada instante em time_array recebe o valor RMS do segmento correspondente,
    considerando que o teste total dura total_time segundos dividido em n_segments.
    """
    segment_length = total_time / n_segments
    emg_signal = np.zeros_like(time_array, dtype=float)
    for i, t in enumerate(time_array):
        seg_idx = int(t // segment_length)
        if seg_idx >= n_segments:
            seg_idx = n_segments - 1
        emg_signal[i] = emg_segments[seg_idx]
    return emg_signal


def synchronize_emg_step(emg_file, quark_file, test_label, n_segments=10):
    """
    Lê o arquivo de EMG tratado (2 linhas: 0 = 80%, 1 = 120%) e o arquivo do Quark,
    gerando a coluna 'EMG_RMS' via step function e adicionando a coluna 'Segment'.
    """
    # Ler arquivo de EMG tratado com conversão de decimais
    df_emg = pd.read_excel(emg_file, decimal=',')
    df_emg.columns = df_emg.columns.str.strip()  # Remove espaços extras
    print("Colunas do arquivo EMG:", df_emg.columns.tolist())

    if test_label == "80%":
        emg_data = df_emg.iloc[0]
    elif test_label == "120%":
        emg_data = df_emg.iloc[1]
    else:
        raise ValueError("test_label deve ser '80%' ou '120%'.")

    emg_rms_segments = []
    for i in range(n_segments):
        col_name = f'RMS Segmento {i + 1}'
        if col_name not in df_emg.columns:
            raise ValueError(f"Coluna '{col_name}' não encontrada. Colunas disponíveis: {df_emg.columns.tolist()}")
        emg_rms_segments.append(emg_data[col_name])
    print(f"\nValores de EMG extraídos para teste {test_label}:", emg_rms_segments)

    # Ler arquivo do Quark
    df_quark = pd.read_excel(quark_file, decimal=',')
    df_quark.columns = df_quark.columns.str.strip()
    df_quark['time_sec'] = pd.to_timedelta(df_quark['t']).dt.total_seconds()
    t_min = df_quark['time_sec'].min()
    df_quark['rel_time_sec'] = df_quark['time_sec'] - t_min
    total_time = df_quark['rel_time_sec'].max()
    print("\nTempo total do teste (s):", total_time)

    df_quark['EMG_RMS'] = create_step_emg(df_quark['rel_time_sec'].values, total_time, emg_rms_segments, n_segments)

    # Adicionar coluna de segmentação (de 0 a n_segments-1)
    segment_length = total_time / n_segments
    df_quark['Segment'] = (df_quark['rel_time_sec'] // segment_length).astype(int)
    df_quark.loc[df_quark['Segment'] >= n_segments, 'Segment'] = n_segments - 1

    print("\nEstatísticas de EMG_RMS:")
    print(df_quark['EMG_RMS'].describe())

    return df_quark, total_time


# ---------- Funções para Análise Estatística Detalhada ----------
def detailed_statistics(df, variables):
    """
    Calcula estatísticas descritivas para as variáveis especificadas.
    Retorna um DataFrame resumo.
    """
    summary_list = []
    for var in variables:
        stats = df[var].describe().to_dict()
        stats['Variable'] = var
        summary_list.append(stats)
    return pd.DataFrame(summary_list)


def analyze_relationships(df, variables):
    """
    Calcula a correlação entre cada variável e EMG_RMS.
    Retorna um DataFrame com a variável e a correlação.
    """
    rel_list = []
    for var in variables:
        if df[var].std() > 0 and df['EMG_RMS'].std() > 0:
            corr = np.corrcoef(df[var], df['EMG_RMS'])[0, 1]
        else:
            corr = np.nan
        rel_list.append({'Variable': var, 'Correlation_EMG': corr})
    return pd.DataFrame(rel_list)


# ---------- Funções para Teste t Pareado e Comparação ----------
def compute_segment_means(df, variables, n_segments=10):
    """
    Agrupa o DataFrame por 'Segment' e calcula a média de cada variável para cada segmento.
    Retorna um dicionário onde cada variável possui um array de n_segments médias.
    """
    grouped = df.groupby('Segment')
    means = {}
    for var in variables:
        means[var] = grouped[var].mean().values
    return means


def perform_paired_t_tests(means80, means120, variables):
    """
    Para cada variável, aplica o teste t pareado entre os 10 segmentos dos testes 80% e 120%.
    Retorna um DataFrame com os t-stat e p-values.
    """
    results = []
    for var in variables:
        data80 = means80[var]
        data120 = means120[var]
        if len(data80) == len(data120) and len(data80) > 0:
            t_stat, p_val = ttest_rel(data80, data120)
        else:
            t_stat, p_val = np.nan, np.nan
        results.append({'Variable': var, 't_stat': t_stat, 'p_value': p_val})
    return pd.DataFrame(results)


# Função para calcular Cohen's d para dados pareados
def cohen_d_paired(x, y):
    diff = x - y
    return np.mean(diff) / np.std(diff, ddof=1)


def comparative_statistics(df80, df120, variables):
    """
    Compara cada variável entre os testes 80% e 120%:
      - Calcula as médias para cada intensidade,
      - Diferença absoluta e percentual,
      - Aplica o teste t pareado utilizando as médias por segmento,
      - Calcula Cohen's d.
    Retorna um DataFrame com estes resultados.
    """
    results = []
    for var in variables:
        mean_80 = df80[var].mean()
        mean_120 = df120[var].mean()
        diff_abs = mean_80 - mean_120
        diff_perc = (diff_abs / mean_80 * 100) if mean_80 != 0 else np.nan

        means80 = df80.groupby('Segment')[var].mean().values
        means120 = df120.groupby('Segment')[var].mean().values

        if len(means80) == len(means120) and len(means80) > 0:
            t_stat, p_val = ttest_rel(means80, means120)
            d_val = cohen_d_paired(means80, means120)
        else:
            t_stat, p_val, d_val = np.nan, np.nan, np.nan

        results.append({
            'Variable': var,
            'Mean_80': mean_80,
            'Mean_120': mean_120,
            'Diff_abs': diff_abs,
            'Diff_perc': diff_perc,
            't_stat': t_stat,
            'p_value': p_val,
            'Cohen_d': d_val
        })
    return pd.DataFrame(results)


# ---------- Função para Gerar Gráficos de Linha Comparativos (em função do tempo normalizado) ----------
def generate_comparison_line_plots(df1, df2, intensity1, intensity2, total_time1, total_time2, variables,
                                   output_folder):
    """
    Para cada variável, gera um gráfico de linha com o eixo x representando o tempo normalizado (0-100%)
    e o eixo y os valores da variável. São plotadas duas linhas: uma para cada intensidade (df1 e df2).
    Os gráficos são salvos na pasta output_folder.
    """
    os.makedirs(output_folder, exist_ok=True)

    # Cria cópias para não modificar os dados originais e calcula tempo normalizado (%)
    df1 = df1.copy()
    df2 = df2.copy()
    df1['norm_time'] = (df1['rel_time_sec'] / total_time1) * 100
    df2['norm_time'] = (df2['rel_time_sec'] / total_time2) * 100

    for var in variables:
        safe_var = sanitize_filename(var)
        plt.figure(figsize=(10, 6))
        plt.plot(df1['norm_time'], df1[var], marker='o', linestyle='-', color='blue', label=f"{intensity1}")
        plt.plot(df2['norm_time'], df2[var], marker='x', linestyle='--', color='red', label=f"{intensity2}")
        plt.xlabel("Tempo Normalizado (%)")
        plt.ylabel(var)
        plt.title(f"Comparação de {var} entre {intensity1} e {intensity2}")
        plt.legend()
        plt.grid(True)
        save_path = os.path.join(output_folder, f"ComparisonLine_{safe_var}.png")
        plt.savefig(save_path)
        plt.close()


# ---------- Funções para Gerar Gráficos (Série Temporal e Barras Comparativas) ----------
def generate_time_series_plots(df, intensity, output_folder):
    """
    Gera gráficos de série temporal para cada variável (exceto tempo, segment e EMG_RMS).
    O eixo x é o tempo relativo e os eixos y exibem, em um eixo, a variável e, no outro, EMG_RMS.
    Os gráficos são salvos na pasta output_folder com nomes sanitizados.
    """
    os.makedirs(output_folder, exist_ok=True)
    skip_cols = {'t', 'time_sec', 'rel_time_sec', 'EMG_RMS', 'Segment'}
    variables = [col for col in df.columns if col not in skip_cols]

    for var in variables:
        safe_var = sanitize_filename(var)
        fig, ax1 = plt.subplots(figsize=(10, 6))
        color1 = 'tab:blue'
        ax1.set_xlabel('Tempo Relativo (s)')
        ax1.set_ylabel(var, color=color1)
        ax1.plot(df['rel_time_sec'], df[var], marker='o', linestyle='-', color=color1, label=var)
        ax1.tick_params(axis='y', labelcolor=color1)

        ax2 = ax1.twinx()
        color2 = 'tab:red'
        ax2.set_ylabel('EMG_RMS', color=color2)
        ax2.plot(df['rel_time_sec'], df['EMG_RMS'], marker='x', linestyle='--', color=color2, label='EMG_RMS')
        ax2.tick_params(axis='y', labelcolor=color2)

        plt.title(f"{intensity} - {var} vs EMG_RMS ao longo do tempo")
        fig.tight_layout()
        save_path = os.path.join(output_folder, f"{sanitize_filename(intensity)}_{safe_var}.png")
        plt.savefig(save_path)
        plt.close()


def generate_comparison_bar_charts(df80, df120, variables, output_folder):
    """
    Gera gráficos de barras comparativos (usando as médias) para cada variável entre Teste 80% e Teste 120%.
    Os gráficos são salvos na pasta output_folder com nomes sanitizados.
    """
    os.makedirs(output_folder, exist_ok=True)
    means80 = df80[variables].mean()
    means120 = df120[variables].mean()
    for var in variables:
        safe_var = sanitize_filename(var)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.bar(['80%', '120%'], [means80[var], means120[var]], color=['blue', 'green'])
        ax.set_title(f"Comparação das Médias de {var}")
        ax.set_ylabel(var)
        save_path = os.path.join(output_folder, f"Comparison_{safe_var}.png")
        plt.savefig(save_path)
        plt.close()


# ---------- Função para Gerar a Interpretação Textual da Comparação (Médias) ----------
def generate_textual_analysis(comp_stats):
    """
    Gera para cada variável uma interpretação textual baseada na comparação entre 80% e 120%.
    Isto inclui médias, diferença absoluta e percentual, teste t e tamanho do efeito.
    Retorna um DataFrame com as colunas "Variable" e "Interpretation".
    """
    interpretations = []
    for _, row in comp_stats.iterrows():
        var = row['Variable']
        mean80 = row['Mean_80']
        mean120 = row['Mean_120']
        diff_abs = row['Diff_abs']
        diff_perc = row['Diff_perc']
        t_stat = row['t_stat']
        p_val = row['p_value']
        d_val = row['Cohen_d']

        # Classificação do tamanho do efeito
        abs_d = abs(d_val)
        if abs_d < 0.2:
            effect = "insignificante"
        elif abs_d < 0.5:
            effect = "pequeno"
        elif abs_d < 0.8:
            effect = "moderado"
        else:
            effect = "grande"

        sig_text = "significativo" if p_val < 0.05 else "não significativo"

        interpretation = (
            f"Para a variável {var}:\n"
            f"- Média em 80%: {mean80:.2f} versus em 120%: {mean120:.2f}, resultando em uma diferença absoluta de {diff_abs:.2f} "
            f"({diff_perc:.2f}%).\n"
            f"- O teste t pareado apresentou t = {t_stat:.2f} com p = {p_val:.3f} ({sig_text}), e o tamanho do efeito (Cohen's d) foi {d_val:.2f} "
            f"(efeito {effect})."
        )
        interpretations.append({'Variable': var, 'Interpretation': interpretation})
    return pd.DataFrame(interpretations)


# ---------- Função para Gerar a Interpretação Textual Comparando Correlações com EMG ----------
def generate_textual_analysis_emg(rel80, rel120):
    """
    Gera uma interpretação textual para cada variável, comparando as correlações com a EMG
    em 80% e 120%.
    Retorna um DataFrame com as colunas "Variable" e "Interpretation_EMG".
    """
    interpretations = []
    for var in rel80['Variable']:
        corr80 = rel80.loc[rel80['Variable'] == var, 'Correlation_EMG'].values[0]
        corr120 = rel120.loc[rel120['Variable'] == var, 'Correlation_EMG'].values[0]

        def classify_corr(corr):
            abs_corr = abs(corr)
            if abs_corr < 0.3:
                return "fraca"
            elif abs_corr < 0.6:
                return "moderada"
            else:
                return "forte"

        level80 = classify_corr(corr80)
        level120 = classify_corr(corr120)

        interpretation = (
            f"Para a variável {var}, a correlação com a EMG foi de {corr80:.2f} ({level80}) em 80% e {corr120:.2f} ({level120}) em 120%."
        )
        interpretations.append({'Variable': var, 'Interpretation_EMG': interpretation})
    return pd.DataFrame(interpretations)


# ---------- Função para Gerar o Relatório Excel Completo com Gráficos e Interpretação ----------
def generate_full_excel_report(df_quark_80, df_quark_120, corr_80, corr_120,
                               stats80, stats120, rel80, rel120, ttest_results, comp_stats,
                               folder_plots_80, folder_plots_120, comp_folder, line_folder,
                               filename='Full_Report.xlsx'):
    """
    Gera um relatório Excel contendo:
      - Planilhas "Teste 80%" e "Teste 120%" com os dados sincronizados.
      - Planilha "Resumo" com estatísticas descritivas, correlações e resultados do teste t pareado.
      - Planilha "Comparativo" com a tabela comparativa detalhada.
      - Planilha "Interpretation" com a interpretação textual da comparação de médias.
      - Planilha "Interpretation_EMG" com a interpretação textual comparando as correlações com a EMG.
      - Planilhas "Gráficos_80", "Gráficos_120", "G_80e120" e "Line_Comparison" inserindo os gráficos salvos.
    """
    textual_df = generate_textual_analysis(comp_stats)
    textual_emg_df = generate_textual_analysis_emg(rel80, rel120)

    with pd.ExcelWriter(filename, engine='xlsxwriter') as writer:
        df_quark_80.to_excel(writer, sheet_name='Teste 80%', index=False)
        df_quark_120.to_excel(writer, sheet_name='Teste 120%', index=False)

        # Resumo estatístico
        stats80['Teste'] = '80%'
        stats120['Teste'] = '120%'
        summary = pd.concat([stats80, stats120], ignore_index=True)
        rel80['Teste'] = '80%'
        rel120['Teste'] = '120%'
        rel_summary = pd.concat([rel80, rel120], ignore_index=True)
        resumo = pd.merge(summary, rel_summary, on=['Teste', 'Variable'], how='outer')
        ttest_results = ttest_results.drop(columns=['Teste'], errors='ignore')
        resumo = pd.merge(resumo, ttest_results, on='Variable', how='left')
        resumo = resumo[['Teste', 'Variable', 'count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max',
                         'Correlation_EMG', 't_stat', 'p_value']]
        resumo.to_excel(writer, sheet_name='Resumo', index=False)

        # Tabela comparativa detalhada
        comp_stats.to_excel(writer, sheet_name='Comparativo', index=False)

        # Inserir as interpretações textuais
        textual_df.to_excel(writer, sheet_name='Interpretation', index=False)
        textual_emg_df.to_excel(writer, sheet_name='Interpretation_EMG', index=False)

        workbook = writer.book

        # Função auxiliar para inserir imagens de uma pasta numa planilha
        def insert_images_from_folder(worksheet, folder, start_row):
            images = [f for f in os.listdir(folder) if f.endswith(".png")]
            row = start_row
            col = 0
            for image in images:
                img_path = os.path.join(folder, image)
                worksheet.insert_image(row, col, img_path, {'x_scale': 0.8, 'y_scale': 0.8})
                row += 20

        worksheet80 = workbook.add_worksheet("Gráficos_80")
        insert_images_from_folder(worksheet80, folder_plots_80, start_row=0)

        worksheet120 = workbook.add_worksheet("Gráficos_120")
        insert_images_from_folder(worksheet120, folder_plots_120, start_row=0)

        worksheetComp = workbook.add_worksheet("G_80e120")
        insert_images_from_folder(worksheetComp, comp_folder, start_row=0)

        worksheetLine = workbook.add_worksheet("Line_Comparison")
        insert_images_from_folder(worksheetLine, line_folder, start_row=0)

    print(f"\nRelatório completo gerado e salvo em '{filename}'.")


# ---------- Função para Gerar Gráficos de Linha Comparativos ----------
def generate_comparison_line_plots(df1, df2, intensity1, intensity2, total_time1, total_time2, variables,
                                   output_folder):
    """
    Para cada variável, gera um gráfico de linha em que o eixo x é o tempo normalizado em % (0 a 100%),
    e o eixo y representa os valores da variável, com linhas para cada intensidade.
    Os gráficos são salvos na pasta output_folder.
    """
    os.makedirs(output_folder, exist_ok=True)
    # Cria cópias e calcula tempo normalizado para cada DataFrame
    df1 = df1.copy()
    df2 = df2.copy()
    df1['norm_time'] = (df1['rel_time_sec'] / total_time1) * 100
    df2['norm_time'] = (df2['rel_time_sec'] / total_time2) * 100

    for var in variables:
        safe_var = sanitize_filename(var)
        plt.figure(figsize=(10, 6))
        plt.plot(df1['norm_time'], df1[var], marker='o', linestyle='-', color='blue', label=f"{intensity1}")
        plt.plot(df2['norm_time'], df2[var], marker='x', linestyle='--', color='red', label=f"{intensity2}")
        plt.xlabel("Tempo Normalizado (%)")
        plt.ylabel(var)
        plt.title(f"Comparação de {var}: {intensity1} vs {intensity2}")
        plt.legend()
        plt.grid(True)
        save_path = os.path.join(output_folder, f"ComparisonLine_{safe_var}.png")
        plt.savefig(save_path)
        plt.close()


# ---------- Função Main ----------
def main():
    # Caminhos dos arquivos (ajuste conforme necessário)
    emg_file = r"C:\Users\RUBENS\PycharmProjects\EMGAnalise\EMGProcessado\resultados.xlsx"
    quark_80_file = r"C:\Users\RUBENS\PycharmProjects\EMGAnalise\QuarkData\CIDO.xlsx"
    quark_120_file = r"C:\Users\RUBENS\PycharmProjects\EMGAnalise\QuarkData\CIDO2.xlsx"

    # Sincronizar os dados dos testes 80% e 120%
    df_quark_80, total_time_80 = synchronize_emg_step(emg_file, quark_80_file, test_label="80%", n_segments=10)
    df_quark_120, total_time_120 = synchronize_emg_step(emg_file, quark_120_file, test_label="120%", n_segments=10)

    # Definir as variáveis para análise (excluindo colunas de tempo, segment e EMG_RMS)
    skip_cols = {'t', 'time_sec', 'rel_time_sec', 'EMG_RMS', 'Segment'}
    variables = [col for col in df_quark_80.columns if col not in skip_cols]

    # Estatísticas descritivas
    stats80 = detailed_statistics(df_quark_80, variables)
    stats120 = detailed_statistics(df_quark_120, variables)

    # Correlações entre cada variável e EMG_RMS
    rel80 = analyze_relationships(df_quark_80, variables)
    rel120 = analyze_relationships(df_quark_120, variables)

    # Gerar gráficos individuais de série temporal
    folder_plots_80 = "Graphs_80"
    folder_plots_120 = "Graphs_120"
    generate_time_series_plots(df_quark_80, "80%", folder_plots_80)
    generate_time_series_plots(df_quark_120, "120%", folder_plots_120)

    # Gerar gráficos comparativos de barras entre os testes (médias)
    folder_comp = "G_80e120"
    generate_comparison_bar_charts(df_quark_80, df_quark_120, variables, folder_comp)

    # Gerar gráficos de linha comparando ambas intensidades (normalizados pelo tempo)
    folder_line = "Line_Comparison"
    generate_comparison_line_plots(df_quark_80, df_quark_120, "80%", "120%", total_time_80, total_time_120, variables,
                                   folder_line)

    # Calcular as médias por segmento para teste t pareado
    means80 = compute_segment_means(df_quark_80, variables, n_segments=10)
    means120 = compute_segment_means(df_quark_120, variables, n_segments=10)
    ttest_results = perform_paired_t_tests(means80, means120, variables)
    ttest_results['Teste'] = 'Pareado'

    print("\nResultados do teste t pareado:")
    print(ttest_results)

    # Calcular comparação detalhada entre intensidades
    comp_stats = comparative_statistics(df_quark_80, df_quark_120, variables)
    print("\nComparação detalhada entre intensidades (80% vs 120%):")
    print(comp_stats)

    # Tabela comparativa simples das médias
    comp_table = pd.DataFrame({
        'Variable': variables,
        'Mean_80': [df_quark_80[var].mean() for var in variables],
        'Mean_120': [df_quark_120[var].mean() for var in variables]
    })
    print("\nComparativo das médias das variáveis (80% vs 120%):")
    print(comp_table)

    # Gerar relatório Excel completo com todos os elementos
    generate_full_excel_report(df_quark_80, df_quark_120,
                               corr_80=np.corrcoef(df_quark_80["VO2"], df_quark_80["EMG_RMS"])[0, 1],
                               corr_120=np.corrcoef(df_quark_120["VO2"], df_quark_120["EMG_RMS"])[0, 1],
                               stats80=stats80, stats120=stats120,
                               rel80=rel80, rel120=rel120,
                               ttest_results=ttest_results,
                               comp_stats=comp_stats,
                               folder_plots_80=folder_plots_80,
                               folder_plots_120=folder_plots_120,
                               comp_folder=folder_comp,
                               line_folder=folder_line,
                               filename='emg_ergometry_report.xlsx')

    # Exemplo de plot simples para visualizar Teste 80%
    plt.figure(figsize=(10, 6))
    plt.plot(df_quark_80["rel_time_sec"], df_quark_80["EMG_RMS"], "o-", label="EMG_RMS (80%)")
    plt.plot(df_quark_80["rel_time_sec"], df_quark_80["VO2"], "x-", label="VO2 (80%)")
    plt.xlabel("Tempo Relativo (s)")
    plt.ylabel("Valor")
    plt.title("Comparação: EMG_RMS vs VO2 - Teste 80%")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    main()
