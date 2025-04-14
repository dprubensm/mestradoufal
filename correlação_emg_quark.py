import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.stats import pearsonr, linregress


def synchronize_data(df_emg_80, df_emg_120, df_quark_80, df_quark_120):
    """
    Sincroniza os dados de EMG (80% e 120%) com os dados do Quark (teste de ergometria 80% e 120%).

    Parâmetros:
      df_emg_80: DataFrame da EMG referente ao teste de 80%
      df_emg_120: DataFrame da EMG referente ao teste de 120%
      df_quark_80: DataFrame do teste de ergometria 80% (deve conter a coluna 't' com o tempo)
      df_quark_120: DataFrame do teste de ergometria 120% (deve conter a coluna 't' com o tempo)

    Retorna:
      df_quark_80 e df_quark_120 com uma coluna adicional 'EMG_signal' sincronizada ao tempo.
    """
    # Remover espaços extras nos nomes das colunas
    df_quark_80.columns = df_quark_80.columns.str.strip()
    df_quark_120.columns = df_quark_120.columns.str.strip()

    # Exibir as colunas para confirmação
    print("Colunas do Quark 80%:", df_quark_80.columns)
    print("Colunas do Quark 120%:", df_quark_120.columns)

    # Verificar se a coluna de tempo 't' existe
    if 't' in df_quark_80.columns:
        quark_time_80 = df_quark_80['t']
    else:
        raise ValueError("A coluna de tempo 't' não foi encontrada no arquivo Quark 80%")

    if 't' in df_quark_120.columns:
        quark_time_120 = df_quark_120['t']
    else:
        raise ValueError("A coluna de tempo 't' não foi encontrada no arquivo Quark 120%")

    # Exibir os primeiros valores da coluna de tempo
    print("Primeiros valores da coluna 't' para Quark 80%:", quark_time_80.head())
    print("Primeiros valores da coluna 't' para Quark 120%:", quark_time_120.head())

    # Converter o tempo para segundos (supondo formato hh:mm:ss)
    quark_time_80 = pd.to_timedelta(quark_time_80, errors='coerce').dt.total_seconds()
    quark_time_120 = pd.to_timedelta(quark_time_120, errors='coerce').dt.total_seconds()

    print("Tempo (segundos) - Quark 80%:", quark_time_80.head())
    print("Tempo (segundos) - Quark 120%:", quark_time_120.head())

    # Para a EMG, usar o índice dos segmentos como base para o tempo.
    # Cria cópia dos DataFrames para evitar SettingWithCopyWarning.
    df_emg_80 = df_emg_80.copy()
    df_emg_120 = df_emg_120.copy()
    segment_time = np.arange(len(df_emg_80)) * 10  # Cada segmento representa 10 segundos (ajuste se necessário)
    df_emg_80.loc[:, 'Time_seconds'] = segment_time
    df_emg_120.loc[:, 'Time_seconds'] = segment_time

    # Interpolação para sincronização (teste de 80%)
    emg_time_80 = df_emg_80['Time_seconds']
    interpolator_80 = interp1d(emg_time_80, df_emg_80['RMS Total'], kind='linear', fill_value='extrapolate')
    emg_interp_80 = interpolator_80(quark_time_80)
    df_quark_80['EMG_signal'] = emg_interp_80

    # Interpolação para sincronização (teste de 120%)
    emg_time_120 = df_emg_120['Time_seconds']
    interpolator_120 = interp1d(emg_time_120, df_emg_120['RMS Total'], kind='linear', fill_value='extrapolate')
    emg_interp_120 = interpolator_120(quark_time_120)
    df_quark_120['EMG_signal'] = emg_interp_120

    return df_quark_80, df_quark_120


def plot_comparison(df_quark_80, df_quark_120):
    """
    Gera gráficos comparando VO2 e EMG_signal para os testes de ergometria de 80% e 120%.

    Parâmetros:
      df_quark_80: DataFrame sincronizado do teste de 80%
      df_quark_120: DataFrame sincronizado do teste de 120%
    """
    plt.figure(figsize=(12, 6))

    # Gráfico para teste de 80%
    plt.subplot(2, 1, 1)
    plt.plot(df_quark_80['t'], df_quark_80['VO2'], label='VO2 (Quark)', color='blue')
    plt.plot(df_quark_80['t'], df_quark_80['EMG_signal'], label='EMG Signal', color='red', alpha=0.7)
    plt.xlabel('Tempo (s)')
    plt.ylabel('Valor')
    plt.title('Comparação de VO2 e EMG (80% Intensidade)')
    plt.legend()

    # Gráfico para teste de 120%
    plt.subplot(2, 1, 2)
    plt.plot(df_quark_120['t'], df_quark_120['VO2'], label='VO2 (Quark)', color='green')
    plt.plot(df_quark_120['t'], df_quark_120['EMG_signal'], label='EMG Signal', color='red', alpha=0.7)
    plt.xlabel('Tempo (s)')
    plt.ylabel('Valor')
    plt.title('Comparação de VO2 e EMG (120% Intensidade)')
    plt.legend()

    plt.tight_layout()
    plt.show()


def calculate_correlation(df_quark_80, df_quark_120):
    """
    Calcula e exibe a correlação entre VO2 e EMG_signal para os testes de 80% e 120%.

    Retorna:
      Tuple contendo os coeficientes de correlação para 80% e 120%.
    """
    correlation_vo2_80 = np.corrcoef(df_quark_80['VO2'], df_quark_80['EMG_signal'])[0, 1]
    correlation_vo2_120 = np.corrcoef(df_quark_120['VO2'], df_quark_120['EMG_signal'])[0, 1]
    print("Correlação VO2-EMG (80%):", round(correlation_vo2_80, 2))
    print("Correlação VO2-EMG (120%):", round(correlation_vo2_120, 2))
    return correlation_vo2_80, correlation_vo2_120


def analyze_variable_relationships(df, test_label):
    """
    Analisa a relação entre o sinal EMG e cada variável do teste de ergometria.

    Para cada variável (exceto as de tempo e a coluna 'EMG_signal'), calcula:
      - Coeficiente de correlação de Pearson e seu p-valor.
      - Regressão linear (inclinação, intercepto, r-value, p-valor e erro padrão).

    Gera também um gráfico de dispersão com a linha de regressão para cada variável, que é salvo como arquivo PNG.

    Parâmetros:
      df: DataFrame com os dados do teste sincronizado.
      test_label: Rótulo para identificar o teste (exemplo: "80%" ou "120%").

    Retorna:
      DataFrame com os resultados estatísticos para cada variável.
    """
    # Exclui as colunas relacionadas ao tempo e à sinal EMG
    variables = [col for col in df.columns if col not in ['EMG_signal', 't', 'Time_seconds']]
    results = []

    for var in variables:
        x = df[var].values
        y = df['EMG_signal'].values

        try:
            corr, p_corr = pearsonr(x, y)
        except Exception as e:
            corr, p_corr = np.nan, np.nan

        try:
            regression = linregress(x, y)
            slope, intercept, r_value, p_reg, std_err = regression
        except Exception as e:
            slope, intercept, r_value, p_reg, std_err = (np.nan,) * 5

        results.append({
            'Variable': var,
            'Pearson_corr': corr,
            'Correlation_pvalue': p_corr,
            'Regression_slope': slope,
            'Regression_intercept': intercept,
            'Regression_rvalue': r_value,
            'Regression_pvalue': p_reg,
            'Regression_std_err': std_err,
        })

        # Gera o gráfico de dispersão com a linha de regressão
        plt.figure(figsize=(8, 6))
        plt.scatter(x, y, label='Dados', alpha=0.7)
        if not np.isnan(slope):
            x_range = np.linspace(np.min(x), np.max(x), 100)
            y_pred = intercept + slope * x_range
            plt.plot(x_range, y_pred, color='red', label='Linha de regressão')
        plt.title(f'Relação entre EMG e {var} ({test_label})')
        plt.xlabel(var)
        plt.ylabel('EMG_signal')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        # Sanitiza o nome da variável para uso no nome do arquivo (substitui "/" por "_")
        safe_var = var.replace("/", "_")
        safe_label = test_label.replace("%", "").strip()
        plot_filename = f'scatter_{safe_var}_{safe_label}.png'
        plt.savefig(plot_filename)
        plt.close()

    df_results = pd.DataFrame(results)
    return df_results


def analyze_emg_ergometry(df_quark_80, df_quark_120):
    """
    Realiza a análise estatística para os testes de ergometria (80% e 120%) e gera uma comparação entre eles.

    Retorna:
      analysis_80: DataFrame com as estatísticas do teste de 80%.
      analysis_120: DataFrame com as estatísticas do teste de 120%.
      comparison: DataFrame com a comparação (diferença) entre os resultados dos dois testes.
    """
    analysis_80 = analyze_variable_relationships(df_quark_80, "80%")
    analysis_120 = analyze_variable_relationships(df_quark_120, "120%")
    comparison = analysis_80.merge(analysis_120, on='Variable', suffixes=('_80', '_120'))
    comparison['Diff_Pearson_corr'] = comparison['Pearson_corr_120'] - comparison['Pearson_corr_80']
    comparison['Diff_Regression_slope'] = comparison['Regression_slope_120'] - comparison['Regression_slope_80']
    return analysis_80, analysis_120, comparison


def export_analysis_to_excel(analysis_80, analysis_120, comparison, output_filename='emg_ergometry_analysis.xlsx'):
    """
    Exporta os resultados das análises para um arquivo Excel com três planilhas:
      - '80% Analysis'
      - '120% Analysis'
      - 'Comparison'

    Parâmetros:
      analysis_80: DataFrame com as estatísticas do teste de 80%
      analysis_120: DataFrame com as estatísticas do teste de 120%
      comparison: DataFrame comparativo dos dois testes
      output_filename: Nome do arquivo Excel de saída.
    """
    with pd.ExcelWriter(output_filename, engine='xlsxwriter') as writer:
        analysis_80.to_excel(writer, sheet_name='80% Analysis', index=False)
        analysis_120.to_excel(writer, sheet_name='120% Analysis', index=False)
        comparison.to_excel(writer, sheet_name='Comparison', index=False)
    print(f"Resultados exportados para '{output_filename}'.")


if __name__ == '__main__':
    # Caminhos para os arquivos de dados (ajuste conforme necessário)
    emg_file_path = r"C:\Users\RUBENS\PycharmProjects\EMGAnalise\EMGProcessado\resultados.xlsx"
    quark_80_file_path = r"C:\Users\RUBENS\PycharmProjects\EMGAnalise\QuarkData\CIDO.xlsx"  # Teste 80%
    quark_120_file_path = r"C:\Users\RUBENS\PycharmProjects\EMGAnalise\QuarkData\CIDO2.xlsx"  # Teste 120%

    # Leitura dos arquivos Excel
    df_emg = pd.read_excel(emg_file_path)
    df_quark_80 = pd.read_excel(quark_80_file_path)
    df_quark_120 = pd.read_excel(quark_120_file_path)

    # Selecione as colunas de interesse para o teste de ergometria
    columns_of_interest = ['Rf', 'mRF', 'VT', 'mVT', 'VE', 'mVE',
                           'VO2', 'mVO2', 'VCO2', 'mVCO2', 'VE/VO2',
                           'mVE/VO2', 'VE/VCO2', 'mVE/VCO2', 'R']
    # Obs.: os arquivos do Quark devem possuir também uma coluna 't' com informações de tempo.
    df_quark_80 = df_quark_80[columns_of_interest + ['t']]
    df_quark_120 = df_quark_120[columns_of_interest + ['t']]

    # Seleciona os testes de EMG (assumindo que a primeira linha é para 80% e a segunda para 120%)
    df_emg_80 = df_emg.iloc[0:1].copy()
    df_emg_120 = df_emg.iloc[1:2].copy()

    # Sincroniza os dados
    df_quark_80_synch, df_quark_120_synch = synchronize_data(df_emg_80, df_emg_120, df_quark_80, df_quark_120)

    # (Opcional) Gerar gráficos comparativos
    # plot_comparison(df_quark_80_synch, df_quark_120_synch)

    # (Opcional) Calcular e exibir correlações
    calculate_correlation(df_quark_80_synch, df_quark_120_synch)

    # Realiza a análise estatística completa e gera os DataFrames de resultados
    analysis_80, analysis_120, comparison = analyze_emg_ergometry(df_quark_80_synch, df_quark_120_synch)

    # Exporta os resultados para um arquivo Excel
    export_analysis_to_excel(analysis_80, analysis_120, comparison, output_filename='emg_ergometry_analysis.xlsx')

    print("Análise estatística e exportação concluídas!")
