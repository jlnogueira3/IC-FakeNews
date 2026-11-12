"""Feedback ponderado com referência temporal e avaliação prequential.

Versão experimental para SGDClassifier(loss="log_loss") em classificação binária.
Treinamento inicial: Fake.br. Fluxo cronológico: FakeRecogna.

A cada dia:
1. Prediz TODAS as notícias do dia antes de atualizar os classificadores.
2. Calcula o peso de feedbacks de treino usando apenas referências rotuladas
   de dias ANTERIORES (janela deslizante).
3. Atualiza os modelos com partial_fit.
4. Adiciona as notícias reservadas para validação à referência temporal.

A referência temporal nunca entra no treino do modelo ponderado.
"""

from __future__ import annotations

import argparse
from collections import deque
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import expit
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier, Perceptron, PassiveAggressiveClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLASSES = np.array([0, 1], dtype=int)  # 0 = verdadeira; 1 = falsa


@dataclass(frozen=True)
class Config:
    """Parâmetros experimentais; manter idênticos nas comparações."""

    inicio: str = "2020-01-01"
    janela_dias: int = 90
    reservar_cada: int = 5  # 1 em cada 5 notícias: validação; nunca treina
    min_validacao: int = 20
    max_validacao: int = 256
    eta: float = 0.05  # learning rate do SGD e do passo virtual
    alpha_meta: float = 10.0  # taxa de aprendizado do peso
    peso_inicial: float = 0.5
    peso_minimo: float = 0.05
    n_features: int = 2**18
    random_state: int = 42

    def validate(self) -> None:
        pd.Timestamp(self.inicio)
        if self.janela_dias < 1 or self.reservar_cada < 2:
            raise ValueError("janela_dias >= 1 e reservar_cada >= 2 são obrigatórios")
        if self.min_validacao < 2 or self.max_validacao < self.min_validacao:
            raise ValueError("max_validacao deve ser >= min_validacao >= 2")
        if not (0 < self.eta <= 1) or self.alpha_meta <= 0:
            raise ValueError("eta deve estar em (0, 1] e alpha_meta > 0")
        if not (0 <= self.peso_minimo <= self.peso_inicial <= 1):
            raise ValueError("Exige 0 <= peso_minimo <= peso_inicial <= 1")


def carregar_fakebr(raiz: Path) -> tuple[list[str], np.ndarray]:
    """Lê a Fake.br sem misturar rótulos das duas bases."""
    pasta = raiz / "database" / "Fake.br" / "corpus" / "full_texts"
    textos: list[str] = []
    rotulos: list[int] = []
    for classe, rotulo in (("true", 0), ("fake", 1)):
        arquivos = sorted((pasta / classe).glob("*.txt"), key=lambda p: p.name)
        if not arquivos:
            raise FileNotFoundError(f"Não há arquivos em {pasta / classe}")
        for arquivo in arquivos:
            texto = arquivo.read_text(encoding="utf-8", errors="replace").strip()
            if texto:
                textos.append(texto)
                rotulos.append(rotulo)
    if len(set(rotulos)) != 2:
        raise ValueError("A Fake.br precisa conter notícias das duas classes")
    return textos, np.array(rotulos, dtype=int)


def converter_data(valor: object) -> pd.Timestamp:
    """Datas brasileiras, ISO e timestamps de planilhas; ignora horários."""
    if pd.isna(valor):
        return pd.NaT
    if isinstance(valor, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(valor).normalize()
    texto = str(valor).strip()
    br = re.search(r"(\d{1,2})/(\d{1,2})/(\d{2,4})", texto)
    if br:
        dia, mes, ano = map(int, br.groups())
        if ano < 100:
            ano += 2000
        try:
            return pd.Timestamp(year=ano, month=mes, day=dia)
        except ValueError:
            return pd.NaT
    iso = re.search(r"(\d{4})[-/](\d{1,2})[-/](\d{1,2})", texto)
    if iso:
        ano, mes, dia = map(int, iso.groups())
        try:
            return pd.Timestamp(year=ano, month=mes, day=dia)
        except ValueError:
            return pd.NaT
    return pd.NaT


def mapear_classe_fakerecogna(valor: object) -> float:
    """A FakeRecogna original usa 0=fake e 1=true (inverso da Fake.br)."""
    if pd.isna(valor):
        return float("nan")
    if isinstance(valor, (bool, np.bool_)):
        return float(0 if valor else 1)
    if isinstance(valor, str):
        rotulo = valor.strip().lower()
        if rotulo in ("true", "verdadeira", "verdadeiro", "real"):
            return 0.0
        if rotulo in ("false", "fake", "falsa", "falso"):
            return 1.0
    try:
        numerico = int(valor)
    except (ValueError, TypeError):
        raise ValueError(f"Rótulo desconhecido da FakeRecogna: {valor!r}") from None
    if numerico not in (0, 1) or str(valor).strip() not in (
        "0", "1", "0.0", "1.0"
    ):
        raise ValueError(f"Rótulo inválido da FakeRecogna: {valor!r}")
    return float(1 - numerico)


def preparar_fakerecogna(df: pd.DataFrame, inicio: str) -> pd.DataFrame:
    """Normaliza colunas e ordena o fluxo sem embaralhar o tempo."""
    obrigatorias = {"Noticia", "Classe", "Data"}
    faltantes = obrigatorias - set(df.columns)
    if faltantes:
        raise ValueError(f"Faltam colunas da FakeRecogna: {sorted(faltantes)}")
    dados = df.copy()
    dados["data"] = dados["Data"].map(converter_data)
    dados["rotulo"] = dados["Classe"].map(mapear_classe_fakerecogna)
    partes = [c for c in ("Titulo", "Subtitulo", "Noticia") if c in dados.columns]
    dados["texto"] = (
        dados[partes].fillna("").astype(str).agg(" ".join, axis=1).str.strip()
    )
    dados = dados.dropna(subset=["data", "rotulo"])
    dados = dados[dados["texto"].str.len() > 0]
    dados = dados[dados["data"] >= pd.Timestamp(inicio)]
    dados = dados.sort_values("data", kind="stable").reset_index(drop=True)
    dados["rotulo"] = dados["rotulo"].astype(int)
    dados["posicao_fluxo"] = np.arange(len(dados), dtype=int)
    return dados[["data", "texto", "rotulo", "posicao_fluxo"]]


def carregar_fakerecogna(raiz: Path, inicio: str) -> pd.DataFrame:
    base = raiz / "database" / "FakeRecogna"
    candidatos = (
        base / "repositorio" / "dataset" / "FakeRecogna_datas_padronizadas.xlsx",
        base / "FakeRecogna.xlsx",
    )
    origem = next((p for p in candidatos if p.is_file()), None)
    if origem is None:
        raise FileNotFoundError("FakeRecogna não encontrada: " + ", ".join(map(str, candidatos)))
    print(f"FakeRecogna: {origem}")
    return preparar_fakerecogna(pd.read_excel(origem), inicio)


def vetorizar_com_intercepto(vetorizador: HashingVectorizer, textos) -> sparse.csr_matrix:
    """Intercepto explícito: permite gradiente exato com SGD(fit_intercept=False)."""
    X = vetorizador.transform(list(textos))
    intercepto = sparse.csr_matrix(np.ones((X.shape[0], 1)))
    return sparse.hstack((X, intercepto), format="csr")


def meta_gradiente(
    modelo: SGDClassifier,
    X_noticia: sparse.csr_matrix,
    y_noticia: int,
    X_validacao: sparse.csr_matrix,
    y_validacao: np.ndarray,
    eta: float,
    peso_inicial: float,
) -> float:
    """d L_val(theta-eta*w*grad_i) / d w, via regra da cadeia.

    Para logistic loss: grad_i=(sigmoid(theta.x_i)-y_i)*x_i.
    Usa gradiente de validação avaliado no passo VIRTUAL, sem atualizar o modelo.
    A multiplicação cruzada inclui o intercepto explícito.
    """
    erro_i = float(expit(modelo.decision_function(X_noticia)[0]) - y_noticia)
    similaridades = (X_validacao @ X_noticia.T).toarray().ravel()
    margens_val = np.asarray(modelo.decision_function(X_validacao), dtype=float)
    margens_provisorias = margens_val - eta * peso_inicial * erro_i * similaridades
    erros_val = expit(margens_provisorias) - y_validacao
    return float(-eta * erro_i * np.mean(erros_val * similaridades))


def metricas_mensais(historico: pd.DataFrame) -> pd.DataFrame:
    if historico.empty:
        raise ValueError("Fluxo temporal vazio: não há notícias para avaliar")
    dados = historico.copy()
    dados["mes"] = pd.to_datetime(dados["data"]).dt.to_period("M").astype(str)
    linhas = []
    # Cada coluna de predição é avaliada sobre o MESMO conjunto de notícias.
    modelos = {
        "ponderado_temporal": "pred_ponderado",
        "imediato_peso_constante": "pred_peso_constante",
        "imediato_todas_noticias": "pred_imediato_total",
    }
    for coluna in dados.columns:
        if coluna.startswith("pred_offline_"):
            modelos["offline_" + coluna[len("pred_offline_"):]] = coluna
        elif coluna.startswith("pred_imediato_") and coluna != "pred_imediato_total":
            modelos["imediato_" + coluna[len("pred_imediato_"):]] = coluna
    for mes, bloco in dados.groupby("mes", sort=True):
        y = bloco["rotulo"].to_numpy()
        for nome, coluna in modelos.items():
            pred = bloco[coluna].to_numpy()
            linhas.append({
                "mes": mes,
                "modelo": nome,
                "noticias": len(bloco),
                "accuracy": accuracy_score(y, pred),
                "f1_macro": f1_score(y, pred, average="macro", labels=[0, 1],
                                     zero_division=0),
                "precision_macro": precision_score(y, pred, average="macro",
                                                    labels=[0, 1], zero_division=0),
                "recall_macro": recall_score(y, pred, average="macro",
                                             labels=[0, 1], zero_division=0),
            })
    return pd.DataFrame(linhas)


def executar_experimento(
    textos_fakebr: list[str],
    rotulos_fakebr: np.ndarray,
    fluxo: pd.DataFrame,
    cfg: Config,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Realiza avaliação temporal antes do treino, por dia de publicação."""
    cfg.validate()
    if fluxo.empty:
        raise ValueError("Não há notícias na FakeRecogna após a data de início")
    if {"Noticia", "Classe", "Data"}.issubset(fluxo.columns):
        fluxo = preparar_fakerecogna(fluxo, cfg.inicio)
    elif {"data", "texto", "rotulo"}.issubset(fluxo.columns):
        # Permite passar diretamente o resultado de carregar_fakerecogna().
        fluxo = fluxo.copy()
        fluxo["data"] = pd.to_datetime(fluxo["data"], errors="coerce").dt.normalize()
        fluxo["rotulo"] = pd.to_numeric(fluxo["rotulo"], errors="coerce")
        fluxo["texto"] = fluxo["texto"].fillna("").astype(str).str.strip()
        fluxo = fluxo.dropna(subset=["data", "rotulo"])
        fluxo = fluxo[
            (fluxo["data"] >= pd.Timestamp(cfg.inicio))
            & (fluxo["texto"].str.len() > 0)
        ].copy()
        if not fluxo["rotulo"].isin([0, 1]).all():
            raise ValueError("O fluxo deve usar 0=verdadeira e 1=falsa")
        fluxo["rotulo"] = fluxo["rotulo"].astype(int)
        fluxo = fluxo.sort_values("data", kind="stable").reset_index(drop=True)
        fluxo["posicao_fluxo"] = np.arange(len(fluxo), dtype=int)
    else:
        raise ValueError("Colunas esperadas: Noticia/Classe/Data ou data/texto/rotulo")
    if fluxo.empty:
        raise ValueError("Nenhuma notícia disponível após a data de início")

    vetorizador = HashingVectorizer(
        lowercase=True, ngram_range=(1, 2), n_features=cfg.n_features,
        alternate_sign=False, norm="l2",
    )
    X_inicial_original = vetorizador.transform(textos_fakebr)
    X_inicial = vetorizar_com_intercepto(vetorizador, textos_fakebr)
    y_inicial = np.asarray(rotulos_fakebr, dtype=int)
    if X_inicial.shape[0] != len(y_inicial) or set(y_inicial) != {0, 1}:
        raise ValueError("Treinamento inicial da Fake.br inválido")

    # Sem penalização e com taxa constante para refletir o passo virtual.
    modelo_base = SGDClassifier(
        loss="log_loss", learning_rate="constant", eta0=cfg.eta,
        penalty=None, fit_intercept=False, average=False,
        random_state=cfg.random_state, max_iter=1000, tol=1e-3,
    )
    modelo_base.fit(X_inicial, y_inicial)
    ponderado = deepcopy(modelo_base)
    constante = deepcopy(modelo_base)
    imediato_total = deepcopy(modelo_base)

    # Baselines do notebook original, sem mudanças de modelo ou vocabulário:
    # default SGD usa hinge (não logistic); preservado para não confundir modelos.
    fabricas = {
        "Perceptron": Perceptron(random_state=cfg.random_state),
        "PassiveAggressiveClassifier": PassiveAggressiveClassifier(random_state=cfg.random_state),
        "MultinomialNB": MultinomialNB(),
        "SGDClassifier": SGDClassifier(random_state=cfg.random_state),
    }
    offline = {
        nome: estimador.fit(X_inicial_original, y_inicial)
        for nome, estimador in fabricas.items()
    }
    imediato = {nome: deepcopy(modelo) for nome, modelo in offline.items()}

    # Cada referência: (data, linha_esparsa, classe).
    # Referências NUNCA entram em partial_fit(ponderado/constante).
    referencias = deque()
    historico = []

    for dia, dados_dia in fluxo.groupby("data", sort=True):
        dia = pd.Timestamp(dia)
        limite = dia - pd.Timedelta(days=cfg.janela_dias)
        while referencias and referencias[0][0] < limite:
            referencias.popleft()

        refs = list(referencias)[-cfg.max_validacao:]
        if len(refs) >= cfg.min_validacao and len({r[2] for r in refs}) == 2:
            X_val = sparse.vstack([r[1] for r in refs], format="csr")
            y_val = np.array([r[2] for r in refs], dtype=int)
        else:
            X_val = None
            y_val = None

        X_dia_original = vetorizador.transform(dados_dia["texto"])
        X_dia = vetorizar_com_intercepto(vetorizador, dados_dia["texto"])
        y_dia = dados_dia["rotulo"].to_numpy(dtype=int)
        posicoes = dados_dia["posicao_fluxo"].to_numpy(dtype=int)
        validacao_reservada = (posicoes % cfg.reservar_cada == 0)

        # Todas as previsões são feitas ANTES de atualizar qualquer modelo no dia.
        pred_ponderado = ponderado.predict(X_dia)
        pred_constante = constante.predict(X_dia)
        pred_total = imediato_total.predict(X_dia)
        pred_offline = {
            nome: modelo.predict(X_dia_original) for nome, modelo in offline.items()
        }
        pred_imediato = {
            nome: modelo.predict(X_dia_original) for nome, modelo in imediato.items()
        }

        for k in range(len(dados_dia)):
            Xi = X_dia[k:k+1]
            yi = int(y_dia[k])
            reserva = bool(validacao_reservada[k])
            grad_meta = np.nan
            peso = np.nan
            if not reserva:
                peso = cfg.peso_inicial
                if X_val is not None:
                    grad_meta = meta_gradiente(
                        ponderado, Xi, yi, X_val, y_val, cfg.eta, cfg.peso_inicial
                    )
                    peso = float(np.clip(
                        cfg.peso_inicial - cfg.alpha_meta * grad_meta,
                        cfg.peso_minimo, 1.0,
                    ))
                ponderado.partial_fit(
                    Xi, [yi], classes=CLASSES, sample_weight=[peso]
                )
                constante.partial_fit(
                    Xi, [yi], classes=CLASSES,
                    sample_weight=[cfg.peso_inicial],
                )
            # Baseline ideal: feedback imediato com TODAS as notícias.
            imediato_total.partial_fit(Xi, [yi], classes=CLASSES, sample_weight=[1.0])

            registro = {
                "data": dia,
                "posicao_fluxo": int(posicoes[k]),
                "rotulo": yi,
                "reserva_validacao": reserva,
                "n_validacao_disponivel": len(refs),
                "n_validacao_usada": 0 if X_val is None else len(y_val),
                "peso_feedback": peso,
                "gradiente_meta": grad_meta,
                "pred_ponderado": int(pred_ponderado[k]),
                "pred_peso_constante": int(pred_constante[k]),
                "pred_imediato_total": int(pred_total[k]),
            }
            for nome in offline:
                registro[f"pred_offline_{nome}"] = int(pred_offline[nome][k])
                registro[f"pred_imediato_{nome}"] = int(pred_imediato[nome][k])
            historico.append(registro)

        # Os quatro modelos imediatos recebem TODO o dia, mas só depois da
        # predição de todas as notícias desse dia (sem vazamento no F1).
        for modelo in imediato.values():
            modelo.partial_fit(X_dia_original, y_dia, classes=CLASSES)

        # A validação do dia atual só poderá ser usada nos dias seguintes.
        for k in np.flatnonzero(validacao_reservada):
            referencias.append((dia, X_dia[k:k+1].copy(), int(y_dia[k])))
        while len(referencias) > cfg.max_validacao:
            referencias.popleft()

    df_historico = pd.DataFrame(historico)
    return df_historico, metricas_mensais(df_historico)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--inicio", default="2020-01-01")
    parser.add_argument("--janela-dias", type=int, default=90)
    parser.add_argument("--reservar-cada", type=int, default=5)
    parser.add_argument("--min-validacao", type=int, default=20)
    parser.add_argument("--max-validacao", type=int, default=256)
    parser.add_argument("--eta", type=float, default=0.05)
    parser.add_argument("--alpha-meta", type=float, default=10.0)
    args = parser.parse_args()
    cfg = Config(
        inicio=args.inicio, janela_dias=args.janela_dias,
        reservar_cada=args.reservar_cada, min_validacao=args.min_validacao,
        max_validacao=args.max_validacao, eta=args.eta, alpha_meta=args.alpha_meta,
    )
    raiz = args.project_root.resolve()
    textos, rotulos = carregar_fakebr(raiz)
    noticias = carregar_fakerecogna(raiz, cfg.inicio)
    print(f"Fake.br: {len(textos)} notícias; FakeRecogna pós-corte: {len(noticias)}")
    historico, mensal = executar_experimento(textos, rotulos, noticias, cfg)
    destino = raiz / "result" / "feedback_ponderado_temporal"
    destino.mkdir(parents=True, exist_ok=True)
    historico.to_csv(destino / "predicoes_e_pesos.csv", index=False)
    mensal.to_csv(destino / "metricas_mensais.csv", index=False)
    print(mensal.pivot(index="mes", columns="modelo", values="f1_macro").round(4))
    diferencas = historico.loc[~historico["reserva_validacao"], "peso_feedback"].dropna()
    print("Peso: média = %.6f, mínimo = %.6f, máximo = %.6f" % (
        diferencas.mean(), diferencas.min(), diferencas.max()
    ))
    print(f"Resultados salvos em: {destino}")


if __name__ == "__main__":
    main()
