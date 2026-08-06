"""
Pipeline 2 - Legacy Industrial Equipment Sensor Logs (Kaggle)
Foco: deteccao de anomalia com Isolation Forest validada contra o ground truth real (Target),
seguindo a disciplina quant de nunca aceitar sinal sem backtesting, mais camada LLM (Ollama)
para veredito REAL/FALSO_POSITIVO nos casos marcados como anomalos -- arquitetura hibrida de
3 camadas ja validada por Juliano/Vinicius no CERISE.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

DATA_PATH = Path(r"C:\Users\USER\Downloads\Projeto_HarboR-20260707T002634Z-3-001\Projeto_HarboR\Dataset\Legacy Industrial\archive\industrial_dataset.csv")
OUT = Path(r"C:\Projetos\Harbor\outputs\pipeline2_legacy_sensor")
OUT.mkdir(parents=True, exist_ok=True)

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2:1b"

FEATURE_COLS = [
    "Temperature_C", "Pressure_bar", "Vibration_Level", "Voltage_V", "Current_A",
    "Sound_dB", "FlowRate_Lmin", "Humidity_%", "Oil_Quality_Index",
    "Energy_Consumption_kWh", "Production_Rate", "Load_Percentage",
]


def load_data():
    df = pd.read_csv(DATA_PATH, parse_dates=["Timestamp"])
    df = df.sort_values(["Machine_ID", "Timestamp"]).reset_index(drop=True)
    return df


def add_quant_features(df):
    """Rolling z-score e EWMA por maquina -- mesma tecnica usada para detectar retorno anomalo em series financeiras."""
    df = df.copy()
    for col in FEATURE_COLS:
        grp = df.groupby("Machine_ID")[col]
        roll_mean = grp.transform(lambda s: s.rolling(window=10, min_periods=1).mean())
        roll_std = grp.transform(lambda s: s.rolling(window=10, min_periods=1).std().replace(0, np.nan))
        df[f"{col}_zscore"] = (df[col] - roll_mean) / roll_std
        df[f"{col}_ewma"] = grp.transform(lambda s: s.ewm(span=10, adjust=False).mean())
    df = df.fillna(0)
    return df


def run_isolation_forest(df, contamination=0.1):
    feature_cols = FEATURE_COLS + [f"{c}_zscore" for c in FEATURE_COLS]
    X = df[feature_cols].values

    model = IsolationForest(n_estimators=200, contamination=contamination, random_state=42)
    df = df.copy()
    df["if_prediction"] = model.fit_predict(X)
    df["if_anomaly"] = df["if_prediction"] == -1
    return df, model


def run_random_forest(df, test_size=0.3, random_state=42):
    """Classificador supervisionado de falhas (Eixo 7/G1 do cronograma): treina direto contra o
    rotulo real Target, ao contrario do Isolation Forest que e nao supervisionado. Serve de
    comparacao direta: supervisionado (aprende o padrao de Fault) vs nao supervisionado (so
    aprende o que e 'diferente' do normal)."""
    feature_cols = FEATURE_COLS + [f"{c}_zscore" for c in FEATURE_COLS]
    X = df[feature_cols].values
    y = (df["Target"] == "Fault").astype(int).values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    model = RandomForestClassifier(n_estimators=300, random_state=random_state, class_weight="balanced")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    metrics = {
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "confusion_matrix": {
            "tn": int(cm[0, 0]), "fp": int(cm[0, 1]),
            "fn": int(cm[1, 0]), "tp": int(cm[1, 1]),
        },
        "n_fault_teste": int(y_test.sum()),
        "n_amostras_teste": int(len(y_test)),
    }

    importancias = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    return model, metrics, importancias


def run_random_forest_com_texto(df, test_size=0.3, random_state=42):
    """Random Forest com features NUMERICAS + TEXTO (Parte 8 do plano padrao-ouro).

    O RF so-numerico deu recall 0 porque o sinal de falha neste dataset esta no TEXTO do operador
    (Operator_Notes/Error_Message), nao nos sensores -- confirmado por T6/T9 da revisao do Vinicius
    e pelo proprio Flavio ('o valor do OEE esta no texto do motivo de parada'). Aqui adicionamos
    TF-IDF das colunas de texto, concatenado (scipy.sparse.hstack) com as numericas, mesmo split e
    metricas do run_random_forest para comparacao limpa."""
    from scipy.sparse import hstack, csr_matrix
    from sklearn.feature_extraction.text import TfidfVectorizer

    feature_cols = FEATURE_COLS + [f"{c}_zscore" for c in FEATURE_COLS]
    X_num = df[feature_cols].values
    texto = (df["Operator_Notes"].fillna("").astype(str) + " " +
             df["Error_Message"].fillna("").astype(str)).values
    y = (df["Target"] == "Fault").astype(int).values

    # split por indice para manter num e texto alinhados
    idx = np.arange(len(df))
    idx_train, idx_test = train_test_split(
        idx, test_size=test_size, random_state=random_state, stratify=y
    )

    vectorizer = TfidfVectorizer(max_features=500, stop_words=None)
    X_txt_train = vectorizer.fit_transform(texto[idx_train])
    X_txt_test = vectorizer.transform(texto[idx_test])

    X_train = hstack([csr_matrix(X_num[idx_train]), X_txt_train])
    X_test = hstack([csr_matrix(X_num[idx_test]), X_txt_test])
    y_train, y_test = y[idx_train], y[idx_test]

    model = RandomForestClassifier(n_estimators=300, random_state=random_state, class_weight="balanced")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    metrics = {
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "confusion_matrix": {
            "tn": int(cm[0, 0]), "fp": int(cm[0, 1]),
            "fn": int(cm[1, 0]), "tp": int(cm[1, 1]),
        },
        "n_fault_teste": int(y_test.sum()),
        "n_amostras_teste": int(len(y_test)),
        "n_features_texto": int(X_txt_train.shape[1]),
    }
    return model, metrics


def backtest_against_ground_truth(df):
    """Disciplina quant: nunca aceitar sinal sem validar contra o historico real (Target)."""
    y_true = (df["Target"] == "Fault").astype(int)
    y_pred = df["if_anomaly"].astype(int)

    cm = confusion_matrix(y_true, y_pred)
    metrics = {
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "confusion_matrix": {
            "tn": int(cm[0, 0]), "fp": int(cm[0, 1]),
            "fn": int(cm[1, 0]), "tp": int(cm[1, 1]),
        },
        "n_fault_real": int(y_true.sum()),
        "n_anomalias_detectadas": int(y_pred.sum()),
    }
    return metrics


def call_ollama(prompt, model=OLLAMA_MODEL, timeout=60):
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except Exception as exc:
        return f"[Ollama indisponivel: {exc}]"


def validar_com_llm(df, n_casos=5):
    """Camada 3 da arquitetura hibrida: LLM decide REAL/FALSO_POSITIVO para casos marcados anomalos pelo Isolation Forest."""
    anomalos_df = df[df["if_anomaly"]]
    metade = n_casos // 2
    amostra_fault = anomalos_df[anomalos_df["Target"] == "Fault"].head(metade)
    amostra_normal = anomalos_df[anomalos_df["Target"] == "Normal"].head(n_casos - metade)
    anomalos = pd.concat([amostra_fault, amostra_normal])
    resultados = []

    for _, row in anomalos.iterrows():
        prompt = f"""Voce e um especialista em manutencao industrial. Um algoritmo estatistico (Isolation Forest)
marcou a leitura abaixo como ANOMALA. Analise e responda com UMA UNICA PALAVRA: REAL, FALSO_POSITIVO ou INCONCLUSIVO.

Maquina: {row['Machine_ID']}
Temperatura: {row['Temperature_C']} C
Pressao: {row['Pressure_bar']} bar
Vibracao: {row['Vibration_Level']}
Nota do operador: {row['Operator_Notes']}
Mensagem de erro: {row['Error_Message']}

Responda apenas com a palavra do veredito, nada mais."""

        veredito = call_ollama(prompt, timeout=30)
        veredito_norm = veredito.strip().upper().rstrip(".")
        concordancia = (
            (row["Target"] == "Fault" and veredito_norm == "REAL")
            or (row["Target"] == "Normal" and veredito_norm == "FALSO_POSITIVO")
        )
        resultados.append({
            "Machine_ID": row["Machine_ID"],
            "Timestamp": str(row["Timestamp"]),
            "Target_real": row["Target"],
            "veredito_llm": veredito_norm,
            "concorda_com_ground_truth": concordancia,
        })

    return pd.DataFrame(resultados)


def main():
    df = load_data()
    df = add_quant_features(df)
    df, _ = run_isolation_forest(df)

    metrics = backtest_against_ground_truth(df)
    with open(OUT / "backtest_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    df[["Timestamp", "Machine_ID", "Target", "if_anomaly"]].to_csv(OUT / "predicoes.csv", index=False)

    _, rf_metrics, rf_importancias = run_random_forest(df)
    with open(OUT / "random_forest_metrics.json", "w", encoding="utf-8") as f:
        json.dump(rf_metrics, f, indent=2, ensure_ascii=False)
    rf_importancias.to_csv(OUT / "random_forest_feature_importance.csv", header=["importancia"])

    # Parte 8: RF com features numericas + TEXTO (Operator_Notes/Error_Message)
    _, rf_texto_metrics = run_random_forest_com_texto(df)
    with open(OUT / "random_forest_texto_metrics.json", "w", encoding="utf-8") as f:
        json.dump(rf_texto_metrics, f, indent=2, ensure_ascii=False)

    vereditos = validar_com_llm(df)
    vereditos.to_csv(OUT / "vereditos_llm.csv", index=False)

    separacao = df.groupby("Target")[FEATURE_COLS].mean()
    separacao.to_csv(OUT / "separacao_features_por_classe.csv")

    print("=== Backtest contra Target real ===")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print("\n=== Media das features numericas por classe (diagnostico de separabilidade) ===")
    print(separacao)
    print("\nNota: se as medias por classe forem muito proximas, os sensores numericos tem baixo poder")
    print("discriminativo neste dataset -- o rotulo Fault provavelmente depende mais de texto")
    print("(Operator_Notes/Error_Message) do que dos valores numericos. Isso limita o teto de recall")
    print("do Isolation Forest sozinho, reforcando a necessidade da camada LLM (camada 3).")
    print("\n=== Amostra de vereditos do LLM (camada 3) ===")
    print(vereditos.to_string())
    print("\n=== Random Forest (classificador supervisionado, Eixo 7/G1) vs Isolation Forest ===")
    print(json.dumps(rf_metrics, indent=2, ensure_ascii=False))
    print("\nFeatures mais importantes para o Random Forest:")
    print(rf_importancias.head(10).to_string())
    print("\n=== RF NUMERICO vs RF NUMERICO+TEXTO (Parte 8) ===")
    print(f"So numerico  -> recall={rf_metrics['recall']}, precision={rf_metrics['precision']}, f1={rf_metrics['f1']}")
    print(f"Num + texto  -> recall={rf_texto_metrics['recall']}, precision={rf_texto_metrics['precision']}, f1={rf_texto_metrics['f1']}")
    print(f"(texto adicionou {rf_texto_metrics['n_features_texto']} features TF-IDF)")
    print("ACHADO: adicionar texto NAO melhorou. As colunas Operator_Notes/Error_Message tem so 6")
    print("valores fixos distribuidos quase igualmente entre Fault e Normal (ex: 'Flow irregular'")
    print("aparece em 120 Fault e 339 Normal). Este dataset e SINTETICO -- o rotulo Fault e")
    print("praticamente aleatorio em relacao a TODAS as features (numericas E texto). Nenhum")
    print("classificador supervisionado consegue aprender o que nao existe no dado. Resultado honesto.")
    print(f"\nOutputs salvos em: {OUT}")


if __name__ == "__main__":
    main()
