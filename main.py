import pandas as pd
import matplotlib.pyplot as plt


# ===================== CARREGAMENTO E PROCESSAMENTO =====================

def carregar_csv(nome_arquivo):
    while True:
        try:
            df = pd.read_csv(nome_arquivo)

            # Verificação de colunas obrigatórias
            if {"timestamp", "consumo_kwh"}.issubset(df.columns):
                return df

            print("\nCSV inválido. O arquivo deve conter as colunas: timestamp, consumo_kwh.")
            nome_arquivo = input("Informe o caminho correto do CSV: ")

        except FileNotFoundError:
            nome_arquivo = input(f"Arquivo '{nome_arquivo}' não encontrado. Informe o caminho correto: ")


def preparar_dados(df):
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['dia'] = df['timestamp'].dt.date
    df['hora'] = df['timestamp'].dt.hour
    df['dia_semana_num'] = df['timestamp'].dt.weekday
    return df



# ============================= ESTATÍSTICAS =============================

def consumo_total_por_dia(df):
    return df.groupby('dia')['consumo_kwh'].sum()


def consumo_medio_por_hora(df):
    return df.groupby('hora')['consumo_kwh'].mean()



# ============================= DESPERDÍCIOS =============================

def detectar_desperdicios(df, limite_madrugada=40, limite_zscore=3):
    media = df['consumo_kwh'].mean()
    desvio = df['consumo_kwh'].std()

    if desvio == 0:
        desvio = 1  # evita divisão por zero

    df['zscore'] = (df['consumo_kwh'] - media) / desvio

    picos = df[df['zscore'] > limite_zscore].assign(tipo='Pico anormal')
    madrugada = df[(df['hora'] <= 5) & (df['consumo_kwh'] > limite_madrugada)].assign(
        tipo=f'Madrugada > {limite_madrugada} kWh'
    )

    alertas = pd.concat([picos, madrugada]).sort_values('timestamp')

    # retorno limpo
    return alertas[['timestamp', 'hora', 'consumo_kwh', 'zscore', 'tipo']]



# =============================== OTIMIZAÇÃO ===============================

def aplicar_otimizacao(df):
    reducoes = {
        "madrugada": 0.25,
        "manha": 0.10,
        "expediente": 0.05,
        "noite": 0.15
    }

    def r(h):
        if h <= 5: return reducoes["madrugada"]
        if h <= 8: return reducoes["manha"]
        if h <= 18: return reducoes["expediente"]
        return reducoes["noite"]

    df = df.copy()
    df['consumo_otimizado'] = df['consumo_kwh'] * (1 - df['hora'].apply(r))
    return df



def calcular_economia(df_o, df_ot):
    o = df_o['consumo_kwh'].sum()
    ot = df_ot['consumo_otimizado'].sum()

    economia = o - ot
    pct = economia / o * 100 if o > 0 else 0

    return o, ot, economia, pct



# =============================== ENERGIA SOLAR ===============================

def simular_solar(df, pct_cover=0.7, sun_hours=4.5, perf=0.75, tarifa=0.8,
                   co2_factor=0.475, custo_kw=5500):

    daily_mean = df.groupby('dia')['consumo_kwh'].sum().mean()
    target = daily_mean * pct_cover

    cap_kw = target / (sun_hours * perf)
    annual_gen = cap_kw * sun_hours * 365 * perf
    economia = annual_gen * tarifa
    co2 = annual_gen * co2_factor
    investimento = cap_kw * custo_kw
    payback = investimento / economia if economia > 0 else 9999

    return {
        "pct_cover": pct_cover,
        "cap_kw": cap_kw,
        "annual_gen": annual_gen,
        "economia": economia,
        "co2": co2,
        "investimento": investimento,
        "payback": payback
    }



def imprimir_relatorio(o, ot, econ, pct, solar):
    print("\n=================== RELATÓRIO FINAL ===================")
    print(f"Consumo original total:       {o:.2f} kWh")
    print(f"Consumo otimizado total:      {ot:.2f} kWh")
    print(f"Economia total obtida:        {econ:.2f} kWh")
    print(f"Redução percentual:           {pct:.2f}%")
    print("\n------ Energia Solar ------")
    print(f"Cobertura solar aplicada:     {int(solar['pct_cover']*100)}%")
    print(f"Capacidade necessária:        {solar['cap_kw']:.2f} kW")
    print(f"Geração anual:                {solar['annual_gen']:.0f} kWh")
    print(f"Economia anual:               R$ {solar['economia']:.2f}")
    print(f"CO₂ evitado:                  {solar['co2']:.0f} kg/ano")
    print(f"Investimento:                 R$ {solar['investimento']:.2f}")
    print(f"Payback:                      {solar['payback']:.1f} anos")
    print("========================================================")



# ================================ GRÁFICOS ================================

def plot_consumo_total_por_dia(df):
    dados = consumo_total_por_dia(df)

    plt.figure(figsize=(12, 5))
    plt.plot(dados.index, dados.values, marker='o')
    plt.xticks(rotation=45)
    plt.title('Consumo Total por Dia (kWh)')
    plt.xlabel('Dia')
    plt.ylabel('Consumo (kWh)')
    plt.tight_layout()
    plt.show()


def plot_consumo_medio_por_hora(df):
    dados = consumo_medio_por_hora(df)

    plt.figure(figsize=(12, 5))
    plt.plot(dados.index, dados.values, marker='o')
    plt.title('Consumo Médio por Hora (kWh)')
    plt.xlabel('Hora')
    plt.ylabel('Consumo Médio (kWh)')
    plt.xticks(range(24))
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_histograma_consumo(df):
    plt.figure(figsize=(10, 5))
    plt.hist(df['consumo_kwh'], bins=30, alpha=0.7)
    plt.title('Distribuição do Consumo (kWh)')
    plt.xlabel('kWh')
    plt.ylabel('Frequência')
    plt.tight_layout()
    plt.show()


def plot_comparacao_consumo(df, consumo_otimizado):
    plt.figure(figsize=(14, 7))

    cor_real = "#1f77b4"
    cor_otim = "#ff7f0e"

    plt.plot(df['timestamp'], df['consumo_kwh'],
             color=cor_real, linewidth=1.5, label="Consumo Real")

    plt.plot(df['timestamp'], consumo_otimizado,
             color=cor_otim, linewidth=1.5, linestyle='--', label="Consumo Otimizado")

    plt.fill_between(
        df['timestamp'],
        df['consumo_kwh'],
        consumo_otimizado,
        where=(df['consumo_kwh'] > consumo_otimizado),
        color=cor_real,
        alpha=0.15,
        interpolate=True,
        label="Economia"
    )

    # Tick de 12 em 12 se possível
    passo = max(1, int(len(df) / 12))
    plt.xticks(df['timestamp'][::passo], rotation=45)

    plt.title("Comparação do Consumo: Real vs Otimizado", fontsize=14)
    plt.xlabel("Tempo", fontsize=12)
    plt.ylabel("Consumo (kWh)", fontsize=12)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()



# =================================== MAIN ===================================

def main():
    df = preparar_dados(carregar_csv('consumo_escritorio.csv'))

    alertas = detectar_desperdicios(df)
    print("\n=== DESPERDÍCIOS ===")
    print(alertas)

    df_ot = aplicar_otimizacao(df)
    o, ot, econ, pct = calcular_economia(df, df_ot)

    while True:
        try:
            cob = float(input("Cobertura solar desejada (0-100): "))
            if 0 <= cob <= 100:
                cob /= 100
                break
            print("Digite um valor entre 0 e 100.")
        except ValueError:
            print("Entrada inválida.")

    solar = simular_solar(df, pct_cover=cob)

    imprimir_relatorio(o, ot, econ, pct, solar)

    # GRÁFICOS
    plot_consumo_total_por_dia(df)
    plot_consumo_medio_por_hora(df)
    plot_histograma_consumo(df)
    plot_comparacao_consumo(df, df_ot['consumo_otimizado'])



if __name__ == "__main__":
    main()
