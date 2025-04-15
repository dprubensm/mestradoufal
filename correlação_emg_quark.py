import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from scipy.stats import ttest_rel, shapiro, wilcoxon
from scipy.interpolate import interp1d
import xlsxwriter

warnings.filterwarnings("ignore", category=RuntimeWarning)

def sanitize_filename(filename):
    filename = str(filename)
    forbidden = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
    for char in forbidden:
        filename = filename.replace(char, '_')
    return filename

def read_quark_clean(quark_file):
    # Lê assumindo que a primeira linha é o cabeçalho real
    df = pd.read_excel(quark_file, decimal=',', header=0)
    df.columns = df.columns.str.strip().astype(str)
    return df

def interpolate_emg_signal(df):
    grouped = df.groupby('Segment')
    seg_times = grouped['rel_time_sec'].mean().values
    emg_vals  = grouped['EMG_RMS'].mean().values
    interp_func = interp1d(seg_times, emg_vals, kind='linear', fill_value="extrapolate")
    df_copy = df.copy()
    df_copy['EMG_RMS_interp'] = interp_func(df_copy['rel_time_sec'].values)
    return df_copy

def create_step_emg(time_array, total_time, emg_segments, n_segments=10):
    segment_length = total_time / n_segments
    emg_signal = np.zeros_like(time_array, dtype=float)
    for i, t in enumerate(time_array):
        seg_idx = int(t // segment_length)
        if seg_idx >= n_segments:
            seg_idx = n_segments - 1
        emg_signal[i] = emg_segments[seg_idx]
    return emg_signal

def synchronize_emg_step(emg_file, quark_file, test_label, n_segments=10):
    # Lê EMG tratado
    df_emg = pd.read_excel(emg_file, decimal=',', header=0)
    df_emg.columns = df_emg.columns.str.strip()
    idx = 0 if test_label == '80%' else 1
    emg_data = df_emg.iloc[idx]
    emg_rms_segments = [emg_data[f'RMS Segmento {i+1}'] for i in range(n_segments)]

    # Lê Quark
    df_quark = read_quark_clean(quark_file)
    df_quark = df_quark.dropna(subset=['t'])
    df_quark['time_sec']     = pd.to_timedelta(df_quark['t']).dt.total_seconds()
    df_quark['rel_time_sec'] = df_quark['time_sec'] - df_quark['time_sec'].min()
    total_time = df_quark['rel_time_sec'].max()

    df_quark['EMG_RMS'] = create_step_emg(
        df_quark['rel_time_sec'].values,
        total_time,
        emg_rms_segments,
        n_segments
    )
    df_quark['Segment'] = (df_quark['rel_time_sec'] // (total_time / n_segments)).astype(int)
    df_quark.loc[df_quark['Segment']>=n_segments,'Segment'] = n_segments-1

    return df_quark, total_time

def detailed_statistics(df, variables):
    rows=[]
    for v in variables:
        stats = df[v].describe().to_dict()
        stats['Variable'] = v
        rows.append(stats)
    return pd.DataFrame(rows)

def analyze_relationships(df, variables):
    rows=[]
    for v in variables:
        corr = np.corrcoef(df[v], df['EMG_RMS'])[0,1] if df[v].std()>0 else np.nan
        rows.append({'Variable':v,'Correlation_EMG':corr})
    return pd.DataFrame(rows)

def compute_segment_means(df, variables):
    g = df.groupby('Segment')
    return {v:g[v].mean().values for v in variables}

def perform_paired_tests(means80, means120, variables):
    rows=[]
    for v in variables:
        diff = means80[v] - means120[v]
        sh_p = shapiro(diff)[1]
        normal = sh_p>=0.05
        if normal:
            t_stat, p_val = ttest_rel(means80[v], means120[v])
            test='t-test'
        else:
            try:
                _, p_val = wilcoxon(means80[v], means120[v])
            except:
                p_val = np.nan
            t_stat=np.nan
            test='Wilcoxon'
        rows.append({
            'Variable':v,'Test':test,'Shapiro_p':sh_p,
            't_stat':t_stat,'p_value':p_val,
            'Normality':('Normal' if normal else 'Non-normal')
        })
    return pd.DataFrame(rows)

def cohen_d_paired(x,y):
    d = x-y
    return d.mean()/d.std(ddof=1)

def comparative_statistics(df80, df120, variables):
    rows=[]
    for v in variables:
        m80, m120 = df80[v].mean(), df120[v].mean()
        diff_abs = m80 - m120
        diff_perc = diff_abs/m80*100 if m80!=0 else np.nan
        seg80 = df80.groupby('Segment')[v].mean().values
        seg120= df120.groupby('Segment')[v].mean().values
        if len(seg80)==len(seg120):
            t_stat, p_val = ttest_rel(seg80,seg120)
            d_val = cohen_d_paired(seg80,seg120)
        else:
            t_stat,p_val,d_val = np.nan,np.nan,np.nan
        rows.append({
            'Variable':v,'Mean_80':m80,'Mean_120':m120,
            'Diff_abs':diff_abs,'Diff_perc':diff_perc,
            't_stat':t_stat,'p_value':p_val,'Cohen_d':d_val
        })
    return pd.DataFrame(rows)

def generate_time_series_plots(df, intensity, output_folder):
    os.makedirs(output_folder,exist_ok=True)
    df_interp = interpolate_emg_signal(df)
    skip={'t','time_sec','rel_time_sec','EMG_RMS','Segment'}
    vars=[c for c in df.columns if c not in skip and isinstance(df[c],pd.Series)]
    for v in vars:
        y=pd.to_numeric(df[v],errors='coerce')
        fig,ax1=plt.subplots(figsize=(10,6))
        ax1.plot(df['rel_time_sec'],y,'o-',color='tab:blue',label=v)
        ax1.set_xlabel('Tempo Relativo (s)'); ax1.set_ylabel(v,color='tab:blue')
        ax2=ax1.twinx()
        ax2.plot(df_interp['rel_time_sec'],df_interp['EMG_RMS_interp'],'x--',color='tab:red',label='EMG_RMS')
        ax2.set_ylabel('EMG_RMS',color='tab:red')
        plt.title(f"{intensity} – {v} vs EMG_RMS"); fig.tight_layout()
        fn=sanitize_filename(f"{intensity}_{v}.png")
        fig.savefig(os.path.join(output_folder,fn)); plt.close()

def generate_comparison_bar_charts(df80, df120, variables, output_folder):
    os.makedirs(output_folder,exist_ok=True)
    m80=df80[variables].mean(); m120=df120[variables].mean()
    for v in variables:
        fig,ax=plt.subplots(figsize=(8,6))
        ax.bar(['80%','120%'],[m80[v],m120[v]],color=['blue','green'])
        ax.set_title(f"Médias de {v}")
        fn=sanitize_filename(f"bar_{v}.png")
        fig.savefig(os.path.join(output_folder,fn)); plt.close()

def generate_comparison_line_plots(df1, df2, i1, i2, t1, t2, variables, output_folder):
    os.makedirs(output_folder,exist_ok=True)
    d1i=interpolate_emg_signal(df1); d2i=interpolate_emg_signal(df2)
    d1i['norm']=d1i['rel_time_sec']/t1*100; d2i['norm']=d2i['rel_time_sec']/t2*100
    for v in variables:
        y1=pd.to_numeric(d1i[v],errors='coerce'); y2=pd.to_numeric(d2i[v],errors='coerce')
        fig,ax=plt.subplots(figsize=(10,6))
        ax.plot(d1i['norm'],y1,'o-',label=i1); ax.plot(d2i['norm'],y2,'x--',label=i2)
        ax.set_xlabel('Tempo Normalizado (%)'); ax.set_ylabel(v); ax.legend(); ax.grid(True)
        fn=sanitize_filename(f"line_{v}.png")
        fig.savefig(os.path.join(output_folder,fn)); plt.close()

def generate_textual_analysis(comp):
    rows=[]
    for _,r in comp.iterrows():
        txt=(f"Variável {r.Variable}: média80={r.Mean_80:.2f}, média120={r.Mean_120:.2f}, "
             f"diff={r.Diff_abs:.2f} ({r.Diff_perc:.2f}%), t={r.t_stat:.2f}, p={r.p_value:.3f}, d={r.Cohen_d:.2f}")
        rows.append({'Variable':r.Variable,'Interpretation':txt})
    return pd.DataFrame(rows)

def generate_textual_analysis_emg(rel80,rel120):
    rows=[]
    for v in rel80.Variable:
        c80=rel80.loc[rel80.Variable==v,'Correlation_EMG'].iloc[0]
        c120=rel120.loc[rel120.Variable==v,'Correlation_EMG'].iloc[0]
        lvl=lambda c:'forte' if abs(c)>0.6 else 'moderada' if abs(c)>0.3 else 'fraca'
        txt=f"{v}: corr80={c80:.2f}({lvl(c80)}), corr120={c120:.2f}({lvl(c120)})"
        rows.append({'Variable':v,'Interpretation_EMG':txt})
    return pd.DataFrame(rows)

def generate_full_excel_report(df80, df120, corr80, corr120,
                               stats80, stats120, rel80, rel120,
                               paired, comp, folder80, folder120, compf, linef,
                               filename='emg_ergometry_report.xlsx'):

    text=generate_textual_analysis(comp)
    text_emg=generate_textual_analysis_emg(rel80,rel120)

    with pd.ExcelWriter(filename, engine='xlsxwriter') as writer:
        df80.to_excel(writer, sheet_name='Teste 80%', index=False)
        df120.to_excel(writer, sheet_name='Teste 120%', index=False)

        stats80['Teste']='80%'; stats120['Teste']='120%'
        summary=pd.concat([stats80,stats120],ignore_index=True)
        rel80['Teste']='80%'; rel120['Teste']='120%'
        rels=pd.concat([rel80,rel120],ignore_index=True)

        resumo=(summary.merge(rels,on=['Teste','Variable'],how='outer')
                  .merge(paired.drop(columns=['Teste'],errors='ignore'),on='Variable',how='left'))
        resumo=resumo[['Teste','Variable','count','mean','std','min','25%','50%','75%','max',
                       'Correlation_EMG','t_stat','p_value','Normality']]
        resumo.to_excel(writer, sheet_name='Resumo', index=False)

        comp.to_excel(writer, sheet_name='Comparativo', index=False)
        text.to_excel(writer, sheet_name='Interpretation', index=False)
        text_emg.to_excel(writer, sheet_name='Interpretation_EMG', index=False)

        book=writer.book
        def ins(ws,folder):
            row=0
            for img in sorted(os.listdir(folder)):
                if img.lower().endswith('.png'):
                    ws.insert_image(row,0,os.path.join(folder,img),{'x_scale':0.7,'y_scale':0.7})
                    row+=20

        ws1=book.add_worksheet('Gráficos_80');     ins(ws1,folder80)
        ws2=book.add_worksheet('Gráficos_120');    ins(ws2,folder120)
        ws3=book.add_worksheet('G_80e120');        ins(ws3,compf)
        ws4=book.add_worksheet('Line_Comparison'); ins(ws4,linef)

    print(f"Relatório salvo em '{filename}'")

def main():
    emg_file=r"C:\Users\RUBENS\PycharmProjects\EMGAnalise\EMGProcessado\resultados.xlsx"
    quark_80=r"C:\Users\RUBENS\PycharmProjects\EMGAnalise\QuarkData\G2.xlsx"
    quark_120=r"C:\Users\RUBENS\PycharmProjects\EMGAnalise\QuarkData\G3.xlsx"

    df80,t80=synchronize_emg_step(emg_file,quark_80,"80%",10)
    df120,t120=synchronize_emg_step(emg_file,quark_120,"120%",10)

    respiratory_vars=['Rf','mRF','VT','mVT','VE','mVE','VO2','mVO2',
                      'VCO2','mVCO2','VE/VO2','mVE/VO2','VE/VCO2','mVE/VCO2','R']
    variables=[v for v in respiratory_vars if v in df80.columns]

    stats80=detailed_statistics(df80,variables)
    stats120=detailed_statistics(df120,variables)

    rel80=analyze_relationships(df80,variables)
    rel120=analyze_relationships(df120,variables)

    folder80="Graphs_80"; folder120="Graphs_120"
    compf="G_80e120"; linef="Line_Comparison"

    generate_time_series_plots(df80,"80%",folder80)
    generate_time_series_plots(df120,"120%",folder120)

    generate_comparison_bar_charts(df80,df120,variables,compf)
    generate_comparison_line_plots(df80,df120,"80%","120%",t80,t120,variables,linef)

    means80=compute_segment_means(df80,variables)
    means120=compute_segment_means(df120,variables)
    paired=perform_paired_tests(means80,means120,variables)
    paired['Teste']='Pareado'

    comp=comparative_statistics(df80,df120,variables)

    generate_full_excel_report(df80,df120,
                               np.corrcoef(df80["VO2"],df80["EMG_RMS"])[0,1],
                               np.corrcoef(df120["VO2"],df120["EMG_RMS"])[0,1],
                               stats80,stats120,rel80,rel120,
                               paired,comp,
                               folder80,folder120,compf,linef,
                               filename='emg_ergometry_report.xlsx')

if __name__=="__main__":
    main()
