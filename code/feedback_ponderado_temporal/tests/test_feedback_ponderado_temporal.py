"""Testes sintéticos da causalidade temporal e do meta-gradiente.

Executar na raiz: python -m pytest code/feedback_ponderado_temporal/tests -q
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import expit
from sklearn.linear_model import SGDClassifier

# O diretório do projeto é chamado "code", nome também usado pela stdlib.
# Importamos este script diretamente para evitar colisões de módulos.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from feedback_ponderado_temporal import (
    Config,
    converter_data,
    executar_experimento,
    mapear_classe_fakerecogna,
    meta_gradiente,
    preparar_fakerecogna,
)


def test_rotulos_das_bases_nao_se_invertem():
    assert mapear_classe_fakerecogna(0) == 1  # FakeRecogna: fake
    assert mapear_classe_fakerecogna(1) == 0  # FakeRecogna: true
    assert mapear_classe_fakerecogna("fake") == 1
    assert mapear_classe_fakerecogna("true") == 0
    assert converter_data("Publicado em 20/05/21 04:30") == pd.Timestamp("2021-05-20")
    assert converter_data(pd.Timestamp("2020-01-02")) == pd.Timestamp("2020-01-02")


def test_preparacao_fakerecogna_filtra_e_ordena_datas():
    df = pd.DataFrame({
        "Noticia": ["mais recente", "mais antiga", "fora do periodo", "invalida"],
        "Classe": [0, 1, 0, 1],
        "Data": ["02/01/2020", "01/01/2020", "31/12/2019", "nao ha"],
    })
    fluxo = preparar_fakerecogna(df, "2020-01-01")
    assert fluxo["texto"].tolist() == ["mais antiga", "mais recente"]
    assert fluxo["rotulo"].tolist() == [0, 1]
    assert fluxo["posicao_fluxo"].tolist() == [0, 1]


class _ModeloComMargemZero:
    def decision_function(self, X):
        return np.zeros(X.shape[0])


def test_sinal_meta_gradiente_depende_da_concordancia():
    X = sparse.csr_matrix([[1.0]])
    modelo = _ModeloComMargemZero()
    alinhado = meta_gradiente(modelo, X, 1, X, np.array([1]), 0.1, 0.5)
    conflitante = meta_gradiente(modelo, X, 1, X, np.array([0]), 0.1, 0.5)
    assert alinhado < 0   # aumenta o peso: w <- w - alpha * gradiente
    assert conflitante > 0  # diminui o peso


def _dados_sinteticos():
    treino = (
        ["vacina estudo evidencia saude"] * 20
        + ["boato desinformacao falso noticia"] * 20
    )
    rotulos = np.array([0] * 20 + [1] * 20)
    dias = pd.date_range("2020-01-01", periods=6, freq="D")
    fluxo = pd.DataFrame({
        "data": [d for d in dias for _ in range(2)],
        "texto": ["vacina evidencia", "boato falso"] * 6,
        "rotulo": [int(j % 2) for j in range(6) for _ in range(2)],
    })
    return treino, rotulos, fluxo


def test_prequential_sem_vazamento_temporal():
    treino, rotulos, fluxo = _dados_sinteticos()
    cfg = Config(
        janela_dias=5, reservar_cada=2, min_validacao=2,
        max_validacao=10, eta=0.02, alpha_meta=10,
        n_features=2**9,
    )
    historico, mensal = executar_experimento(treino, rotulos, fluxo, cfg)
    assert len(historico) == len(fluxo)
    assert mensal["modelo"].nunique() == 11
    assert historico["posicao_fluxo"].is_unique
    # Predição do primeiro dia deve ser idêntica no offline e imediato,
    # porque nenhuma atualização no fluxo ainda ocorreu.
    for nome in ("Perceptron", "PassiveAggressiveClassifier", "MultinomialNB", "SGDClassifier"):
        assert (
            historico.iloc[:2][f"pred_offline_{nome}"].tolist()
            == historico.iloc[:2][f"pred_imediato_{nome}"].tolist()
        )
    assert historico["reserva_validacao"].sum() == 6
    assert historico.loc[historico["reserva_validacao"], "peso_feedback"].isna().all()
    assert (historico.loc[~historico["reserva_validacao"], "peso_feedback"] >= 0.05).all()

    # No dia 1 a referência está vazia; no dia 2 há apenas uma classe.
    assert (historico.iloc[0:2]["n_validacao_usada"] == 0).all()
    assert (historico.iloc[2:4]["n_validacao_usada"] == 0).all()
    assert (historico.iloc[4:6]["n_validacao_usada"] == 2).all()

    # A edição de rótulos futuros não pode modificar métricas/pesos/predições anteriores.
    adulterado = fluxo.copy()
    adulterado.loc[adulterado["data"] == pd.Timestamp("2020-01-06"), "rotulo"] = 1
    historico_adulterado, _ = executar_experimento(treino, rotulos, adulterado, cfg)
    pd.testing.assert_frame_equal(
        historico.iloc[:10].reset_index(drop=True),
        historico_adulterado.iloc[:10].reset_index(drop=True),
    )


def test_validacao_expira_depois_da_janela():
    treino, rotulos, fluxo = _dados_sinteticos()
    cfg = Config(
        janela_dias=1, reservar_cada=2,
        min_validacao=2, max_validacao=10, n_features=2**9,
    )
    hist, _ = executar_experimento(treino, rotulos, fluxo, cfg)
    assert (hist["n_validacao_usada"] == 0).all()


def test_derivada_analitica_confere_com_diferencas_finitas():
    X = sparse.csr_matrix([[1.0, 1.0], [-1.0, 1.0]])
    y = np.array([1, 0])
    modelo = SGDClassifier(
        loss="log_loss", penalty=None, learning_rate="constant",
        eta0=0.1, fit_intercept=False, random_state=42,
    )
    modelo.partial_fit(X, y, classes=[0, 1])
    noticia = sparse.csr_matrix([[0.75, 1.0]])
    yi, eta, peso = 1, 0.1, 0.5
    grad = meta_gradiente(modelo, noticia, yi, X, y, eta, peso)
    coef = modelo.coef_.copy()[0]
    erro = float(expit(modelo.decision_function(noticia)[0]) - yi)
    gi = erro * noticia.toarray().ravel()

    def loss_virtual(w):
        logits = X @ (coef - eta * w * gi)
        return np.mean(np.logaddexp(0, logits) - y * logits)

    delta = 1e-5
    grad_numerico = (loss_virtual(peso + delta) - loss_virtual(peso - delta)) / (2 * delta)
    assert np.isclose(grad, grad_numerico, atol=1e-8)
