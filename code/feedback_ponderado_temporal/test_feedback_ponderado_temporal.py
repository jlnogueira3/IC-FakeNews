"""Testes sintéticos: sem dependência das bases reais."""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import expit
from sklearn.linear_model import SGDClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
from feedback_ponderado_temporal import (
    Config, converter_data, mapear_classe_fakerecogna, preparar_fakerecogna,
    meta_gradiente, executar_experimento,
)


class TesteFeedbackTemporal(unittest.TestCase):
    def test_rotulos_invertidos_e_datas(self):
        self.assertEqual(mapear_classe_fakerecogna(0), 1)
        self.assertEqual(mapear_classe_fakerecogna(1), 0)
        self.assertEqual(mapear_classe_fakerecogna("fake"), 1)
        self.assertEqual(mapear_classe_fakerecogna("true"), 0)
        self.assertEqual(converter_data("Publicado em 20/05/21 04:30"), pd.Timestamp("2021-05-20"))
        with self.assertRaises(ValueError):
            mapear_classe_fakerecogna(3)

    def test_gradient_check_por_diferencas_finitas(self):
        X = sparse.csr_matrix([[1.0, 1.0], [-1.0, 1.0]])
        y = np.array([1, 0])
        modelo = SGDClassifier(
            loss="log_loss", penalty=None, learning_rate="constant",
            eta0=0.1, fit_intercept=False, random_state=42,
        )
        modelo.partial_fit(X, y, classes=[0, 1])
        noticia = sparse.csr_matrix([[0.75, 1.0]])
        y_noticia = 1
        alpha = 0.1
        w = 0.5
        analitico = meta_gradiente(modelo, noticia, y_noticia, X, y, alpha, w)

        coef = modelo.coef_.copy()
        erro = float(expit(modelo.decision_function(noticia)[0]) - y_noticia)
        grad_i = erro * noticia.toarray()[0]

        def perda_val(peso):
            margens = (X @ (coef[0] - alpha * peso * grad_i)).ravel()
            return np.mean(np.logaddexp(0, margens) - y * margens)

        eps = 1e-5
        numerico = (perda_val(w + eps) - perda_val(w - eps)) / (2 * eps)
        self.assertAlmostEqual(analitico, numerico, places=6)

    def test_sem_vazamento_mesmo_dia_e_separacao_da_validacao(self):
        # Duas notícias por dia, uma reservada para a referência.
        datas = pd.date_range("2020-01-01", periods=10, freq="D")
        linhas = []
        for data in datas:
            linhas.extend([
                {"Data": data, "Noticia": "falso boato fraude", "Classe": 0},
                {"Data": data, "Noticia": "verdade oficial comprovada", "Classe": 1},
            ])
        fluxo = preparar_fakerecogna(pd.DataFrame(linhas), "2020-01-01")
        cfg = Config(
            reservar_cada=2, min_validacao=2, max_validacao=10,
            janela_dias=7, n_features=64, eta=0.01,
        )
        historico, mensal = executar_experimento(
            ["falso boato", "verdade oficial", "fraude falsa", "fonte verdadeira"],
            np.array([1, 0, 1, 0]), fluxo, cfg
        )
        self.assertEqual(len(historico), len(fluxo))
        self.assertFalse(mensal.empty)
        self.assertEqual(historico["reserva_validacao"].sum(), 10)
        self.assertTrue(historico.loc[historico.reserva_validacao, "peso_feedback"].isna().all())
        self.assertTrue(historico.loc[~historico.reserva_validacao, "peso_feedback"].between(0.05, 1).all())
        self.assertEqual(
            historico.loc[historico.data == pd.Timestamp("2020-01-01"), "n_validacao_disponivel"].max(),
            0,
        )
        # No dia 2, apenas as notícias reservadas no dia 1 são referências.
        self.assertEqual(
            historico.loc[historico.data == pd.Timestamp("2020-01-02"), "n_validacao_disponivel"].max(),
            1,
        )
        self.assertTrue(set(historico["rotulo"]) == {0, 1})


if __name__ == "__main__":
    unittest.main()
