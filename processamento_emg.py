import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, detrend, find_peaks
from numpy.fft import fft

# ------------------------------------------------------------------------------
# Função para remover outliers usando o método dos quartis
# ------------------------------------------------------------------------------
def outliers(X, method='quartile', alpha=7):
    X = np.array(X)
    if method == 'quartile':
        Q1 = np.percentile(X, 25)
        Q3 = np.percentile(X, 76)  # seguindo o código original
        rangedown = Q1 - alpha * (Q3 - Q1)
        rangeup = Q3 + alpha * (Q3 - Q1)
        Y = X[(X > rangedown) & (X < rangeup)]
        out = X[(X <= rangedown) | (X >= rangeup)]
        return Y, out
    else:
        raise ValueError("Método não implementado.")

# ------------------------------------------------------------------------------
# Função similar à 'fakewavelets' do MATLAB para decomposição em bandas
# ------------------------------------------------------------------------------
def fakewavelets(muscle, SR):
    # Remove o componente DC (tendência linear)
    muscle = detrend(muscle)
    # Definindo as bandas de frequência (9 bandas) conforme o MATLAB
    max_freqs = np.array([48.45, 75.75, 110, 149, 193.45, 244.45, 300.8, 363.8, 431.65])
    min_freqs = np.array([26.95, 48.45, 74.8, 108, 146.95, 191.75, 242.2, 297.4, 359.35])

    musclefilt = np.zeros((len(muscle), len(max_freqs)))
    musclefilt2 = np.zeros(len(max_freqs))

    for i in range(len(max_freqs)):
        low = min_freqs[i] / (SR / 2)
        high = max_freqs[i] / (SR / 2)
        b, a = butter(5, [low, high], btype='bandpass')
        filtered = filtfilt(b, a, muscle)
        musclefilt[:, i] = filtered
        musclefilt2[i] = np.mean(np.abs(filtered))

    overall = np.sum(musclefilt2)
    highfreq = np.sum(musclefilt2[4:7])  # bandas 5 a 7 (índices 4, 5, 6)
    lowfreq = np.sum(musclefilt2[0:2])    # bandas 1 e 2 (índices 0 e 1)

    return musclefilt2, musclefilt, highfreq, lowfreq, overall

# ------------------------------------------------------------------------------
# Função para importar dados do arquivo Excel (equivalente ao "importador" do MATLAB)
# ------------------------------------------------------------------------------
def importador(arquivo):
    # Lê o arquivo Excel ignorando a primeira linha de cabeçalho (skiprows=4)
    df = pd.read_excel(arquivo, skiprows=4)
    # Converte todas as colunas para string, substitui vírgula por ponto e converte para numérico
    df = df.apply(lambda col: pd.to_numeric(col.astype(str).str.replace(',', '.'), errors='coerce'))
    return df.values  # converte para numpy array

# ------------------------------------------------------------------------------
# Função principal para processar os dados EMG (equivalente ao emgciclismo4_PIBIC)
# ------------------------------------------------------------------------------
def process_emg_data(filenames, freq=2500):
    tudo = []    # Armazena os sinais (ex.: canal 1) de cada arquivo
    listas = []  # Armazena o nome (sem extensão) de cada teste
    plt.figure()
    for file in filenames:
        dados = importador(file)
        listas.append(os.path.splitext(file)[0])
        # Supõe-se que a primeira coluna é tempo e a segunda é o sinal EMG
        tempo = np.array(dados[:, 0], dtype=float)
        sinal = np.array(dados[:, 1], dtype=float)
        tudo.append(sinal)
        plt.plot(tempo, sinal, label=os.path.basename(file))
    plt.title("Sinais EMG - Canal 1")
    plt.xlabel("Tempo")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.show()

    # Cálculo do fator de normalização baseado nos picos de todos os sinais
    picos_total = []
    for sinal in tudo:
        sinal_detrended = detrend(sinal)
        # Filtragem Butterworth entre 20 e 450 Hz
        b, a = butter(3, [20 / (freq / 2), 450 / (freq / 2)], btype='bandpass')
        sinal_filtrado = filtfilt(b, a, sinal_detrended)
        sinal_abs = np.abs(sinal_filtrado)
        sinal_limpo, _ = outliers(sinal_abs, method='quartile', alpha=7)
        # Detecta picos (usando find_peaks do scipy)
        peaks, _ = find_peaks(sinal_limpo)
        if len(peaks) > 0:
            # Seleciona os 100 maiores picos (se existirem)
            picos = np.sort(sinal_limpo[peaks])[-100:]
            picos_total.extend(picos)
    picos_total = np.sort(np.array(picos_total))[::-1]
    if len(picos_total) >= 30:
        normfatorC1 = np.mean(picos_total[:30])
    else:
        normfatorC1 = 1  # valor padrão se não houver picos suficientes

    # Cálculo de RMS e frequência mediana para cada sinal
    rms_total = []
    Fm_total = []
    rms_10porcento = []
    Fm_10porcento = []

    for sinal in tudo:
        sinal = detrend(sinal)
        b, a = butter(3, [20 / (freq / 2), 450 / (freq / 2)], btype='bandpass')
        sinal_filtrado = filtfilt(b, a, sinal)
        sinal_norm = sinal_filtrado / normfatorC1
        n = len(sinal_norm)

        # Divisão do sinal em 10 partes aproximadamente iguais
        indices = [int(i * n / 10) for i in range(11)]
        rms_seg = []
        fm_seg = []
        for i in range(10):
            segment = sinal_norm[indices[i]:indices[i + 1]]
            # Cálculo do RMS para o segmento
            rms_val = np.sqrt(np.mean(segment ** 2)) * 100
            rms_seg.append(rms_val)
            # Cálculo da frequência mediana via FFT
            N_seg = len(segment)
            Y = fft(segment)
            PSY = (np.abs(Y) ** 2) / N_seg
            f_axis = np.linspace(0, freq, N_seg, endpoint=False)[:N_seg // 2]
            total_power = np.trapz(PSY[:N_seg // 2], f_axis)
            cumulative = 0
            fm_val = 0
            for j, p in enumerate(PSY[:N_seg // 2]):
                cumulative += p
                if cumulative >= total_power * 0.5:
                    fm_val = f_axis[j]
                    break
            fm_seg.append(fm_val)
        rms_total_val = np.sqrt(np.mean(sinal_norm ** 2)) * 100
        rms_total.append(rms_total_val)

        # Cálculo da frequência mediana para o sinal total
        N_total = len(sinal_norm)
        Y_total = fft(sinal_norm)
        PSY_total = (np.abs(Y_total) ** 2) / N_total
        f_axis_total = np.linspace(0, freq, N_total, endpoint=False)[:N_total // 2]
        total_power = np.trapz(PSY_total[:N_total // 2], f_axis_total)
        cumulative = 0
        fm_total_val = 0
        for j, p in enumerate(PSY_total[:N_total // 2]):
            cumulative += p
            if cumulative >= total_power * 0.5:
                fm_total_val = f_axis_total[j]
                break
        Fm_total.append(fm_total_val)
        rms_10porcento.append(rms_seg)
        Fm_10porcento.append(fm_seg)

    # Processa cada sinal com fakewavelets para obter as bandas de frequência
    musclefilt2_C1_list = []
    highfreq_list = []
    lowfreq_list = []
    overall_list = []
    for sinal in tudo:
        musclefilt2, _, highfreq, lowfreq, overall = fakewavelets(sinal, freq)
        musclefilt2_C1_list.append(musclefilt2)
        highfreq_list.append(highfreq)
        lowfreq_list.append(lowfreq)
        overall_list.append(overall)

    resultados = {
        "rms_10porcento": rms_10porcento,
        "rms_total": rms_total,
        "Fm_10porcento": Fm_10porcento,
        "Fm_total": Fm_total
    }

    return (resultados, tudo, listas, normfatorC1,
            musclefilt2_C1_list, highfreq_list, lowfreq_list, overall_list)

# ------------------------------------------------------------------------------
# Chama a função para rodar em main.py
# ------------------------------------------------------------------------------
if __name__ == '__main__':
    # Lista de arquivos Excel com dados de EMG (modifique conforme necessário)
    arquivos_emg = ["planilhas/80gabriel.xlsx", "planilhas/120gabriel.xlsx"]
    resultados, sinais, lista_nomes, normfator, musclefilt2, highfreq, lowfreq, overall = process_emg_data(arquivos_emg)

    print("Fator de normalização:", normfator)
    print("Resultados (RMS e Frequência mediana):", resultados)

    # ------------------------------------------------------------------------------
    # Chama a função para rodar em main.py
    # ------------------------------------------------------------------------------
    import pandas as pd

    data = []
    for i, teste in enumerate(lista_nomes):
        entry = {}
        entry['Teste'] = teste
        entry['RMS Total'] = resultados['rms_total'][i]
        entry['Fm Total'] = resultados['Fm_total'][i]
        # Salvando os valores dos 10 segmentos
        for j in range(10):
            entry[f'RMS Segmento {j + 1}'] = resultados['rms_10porcento'][i][j]
            entry[f'Fm Segmento {j + 1}'] = resultados['Fm_10porcento'][i][j]
        data.append(entry)

    df_resultados = pd.DataFrame(data)
    df_resultados.to_excel("resultados.xlsx", index=False)
    print("Os resultados foram salvos em 'resultados.xlsx'")

