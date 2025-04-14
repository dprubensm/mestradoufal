import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

# Suprime warnings de RuntimeWarning (caso venham de outras chamadas)
warnings.filterwarnings("ignore", category=RuntimeWarning)


def create_step_emg(time_array, total_time, emg_segments, n_segments=10):
    """
    Cria uma step function para os dados de EMG.

    Cada ponto de tempo em time_array recebe o valor de RMS do segmento correspondente,
    considerando que o teste total dura total_time segundos dividido em n_segments iguais.
    """
    segment_length = total_time / n_segments
    emg_signal = np.zeros_like(time_array, dtype=float)

    for i, t in enumerate(time_array):
        seg_idx = int(t // segment_length)
        if seg_idx >= n_segments:
            seg_idx = n_segments - 1  # Evita extrapolação
        emg_signal[i] = emg_segments[seg_idx]
    return emg_signal


def synchronize_emg_step(emg_file, quark_file, test_label, n_segments=10):
    """
    Lê o arquivo de EMG tratado (2 linhas: linha 0 = 80%, linha 1 = 120%) e o arquivo do Quark,
    e cria uma nova coluna 'EMG_RMS' no DataFrame do Quark associando, via step function,
    os 10 valores de RMS (um para cada segmento) ao tempo dos dados do Quark.
    """
    # 1) Ler o arquivo de EMG tratado com conversão correta dos decimais
    df_emg = pd.read_excel(emg_file, decimal=',')
    df_emg.columns = df_emg.columns.str.strip()  # Remove espaços extras
    print("Colunas do arquivo EMG:", df_emg.columns.tolist())
    print("Tipos das colunas de EMG:", df_emg.dtypes.tolist())

    # Seleciona a linha desejada: linha 0 para 80% e linha 1 para 120%
    if test_label == "80%":
        emg_data = df_emg.iloc[0]
    elif test_label == "120%":
        emg_data = df_emg.iloc[1]
    else:
        raise ValueError("test_label deve ser '80%' ou '120%'.")

    # 2) Extrai os 10 valores de RMS dos segmentos
    emg_rms_segments = []
    for i in range(n_segments):
        col_name = f'RMS Segmento {i + 1}'
        if col_name not in df_emg.columns:
            raise ValueError(
                f"Coluna '{col_name}' não encontrada no arquivo de EMG. " +
                f"Colunas disponíveis: {df_emg.columns.tolist()}"
            )
        emg_value = emg_data[col_name]
        emg_rms_segments.append(emg_value)

    print(f"\nValores de EMG extraídos para teste {test_label}:")
    print(emg_rms_segments)

    # 3) Ler o arquivo do Quark
    df_quark = pd.read_excel(quark_file, decimal=',')
    df_quark.columns = df_quark.columns.str.strip()

    # Converter a coluna 't' para segundos
    df_quark['time_sec'] = pd.to_timedelta(df_quark['t']).dt.total_seconds()
    t_min = df_quark['time_sec'].min()
    df_quark['rel_time_sec'] = df_quark['time_sec'] - t_min
    total_time = df_quark['rel_time_sec'].max()
    print("\nTempo total do teste (s):", total_time)

    # 4) Cria a coluna 'EMG_RMS' via step function
    df_quark['EMG_RMS'] = create_step_emg(df_quark['rel_time_sec'].values, total_time, emg_rms_segments, n_segments)

    # Debug: mostrar estatísticas de EMG_RMS
    print("\nEstatísticas de EMG_RMS:")
    print(df_quark['EMG_RMS'].describe())

    return df_quark, total_time


def main():
    # Defina os caminhos dos arquivos
    emg_file = r"C:\Users\RUBENS\PycharmProjects\EMGAnalise\EMGProcessado\resultados.xlsx"
    quark_80_file = r"C:\Users\RUBENS\PycharmProjects\EMGAnalise\QuarkData\CIDO.xlsx"
    quark_120_file = r"C:\Users\RUBENS\PycharmProjects\EMGAnalise\QuarkData\CIDO2.xlsx"

    # Sincroniza para os testes 80% e 120%
    df_quark_80, total_time_80 = synchronize_emg_step(emg_file, quark_80_file, test_label="80%", n_segments=10)
    df_quark_120, total_time_120 = synchronize_emg_step(emg_file, quark_120_file, test_label="120%", n_segments=10)

    # Debug: Verificar estatísticas de VO2
    print("\nEstatísticas da coluna VO2 (80%):")
    print(df_quark_80["VO2"].describe())
    print("\nEstatísticas da coluna VO2 (120%):")
    print(df_quark_120["VO2"].describe())

    # Verificar se há NaN
    print("\nNúmero de NaN em VO2 (80%):", df_quark_80["VO2"].isna().sum())
    print("Número de NaN em EMG_RMS (80%):", df_quark_80["EMG_RMS"].isna().sum())

    # Cálculo da correlação (apenas se ambas as colunas tiverem variância)
    if df_quark_80["VO2"].std() > 0 and df_quark_80["EMG_RMS"].std() > 0:
        corr_80 = np.corrcoef(df_quark_80["VO2"], df_quark_80["EMG_RMS"])[0, 1]
        print(f"\nCorrelação VO2 x EMG_RMS (80%): {corr_80:.4f}")
    else:
        print("\n[80%] Desvio padrão de VO2 ou EMG_RMS é zero; correlação não definida.")

    if df_quark_120["VO2"].std() > 0 and df_quark_120["EMG_RMS"].std() > 0:
        corr_120 = np.corrcoef(df_quark_120["VO2"], df_quark_120["EMG_RMS"])[0, 1]
        print(f"Correlação VO2 x EMG_RMS (120%): {corr_120:.4f}")
    else:
        print("[120%] Desvio padrão de VO2 ou EMG_RMS é zero; correlação não definida.")

    # Plot de exemplo para o teste de 80%
    plt.figure(figsize=(10, 6))
    plt.plot(df_quark_80["rel_time_sec"], df_quark_80["EMG_RMS"], "o-", label="EMG_RMS (80%)")
    if "VO2" in df_quark_80.columns:
        plt.plot(df_quark_80["rel_time_sec"], df_quark_80["VO2"], "x-", label="VO2 (80%)")
    plt.xlabel("Tempo Relativo (s)")
    plt.ylabel("Valor")
    plt.title("Comparação: EMG_RMS (Step Function) vs VO2 - Teste 80%")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Exporta os DataFrames sincronizados para Excel
    df_quark_80.to_excel("quark_80_sync.xlsx", index=False)
    df_quark_120.to_excel("quark_120_sync.xlsx", index=False)
    print("\nArquivos 'quark_80_sync.xlsx' e 'quark_120_sync.xlsx' gerados com sucesso!")


if __name__ == '__main__':
    main()
