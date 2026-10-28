from pathlib import Path
import re

import pandas as pd


# ============================================================
# CONFIGURAÇÕES
# ============================================================

# Diretório em que este script está localizado:
# Codigos/code/preparacao/prepare_temporal_split.py
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]

# Estrutura esperada:
# Codigos/database/Fake.br/corpus_copia_original/full_texts/
DATASET_DIR = PROJECT_ROOT / "database" / "Fake.br" / "corpus_copia_original"
FULL_TEXTS_DIR = DATASET_DIR / "full_texts"

# Diretório em que os arquivos processados serão salvos
OUTPUT_DIR = PROJECT_ROOT / "database" / "Fake.br" / "processados" / "divisao_temporal"

# Treino: notícias anteriores a 01/12/2019
# Teste: notícias a partir de 01/12/2019
CUTOFF_DATE = pd.Timestamp("2019-12-01")


# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================

def validate_directory_structure() -> None:
    """
    Verifica se todas as pastas necessárias existem.
    """

    required_directories = [
        FULL_TEXTS_DIR / "fake",
        FULL_TEXTS_DIR / "fake-meta-information",
        FULL_TEXTS_DIR / "true",
        FULL_TEXTS_DIR / "true-meta-information",
    ]

    print("=" * 70)
    print("VERIFICAÇÃO DA ESTRUTURA DO CORPUS")
    print("=" * 70)
    print(f"Diretório do script: {SCRIPT_DIR}")
    print(f"Diretório do corpus: {DATASET_DIR}")
    print(f"Diretório full_texts: {FULL_TEXTS_DIR}")
    print()

    if not DATASET_DIR.exists():
        raise FileNotFoundError(
            "A pasta Fake.br-Corpus não foi encontrada.\n"
            f"Caminho procurado: {DATASET_DIR}"
        )

    if not FULL_TEXTS_DIR.exists():
        raise FileNotFoundError(
            "A pasta full_texts não foi encontrada.\n"
            f"Caminho procurado: {FULL_TEXTS_DIR}"
        )

    for directory in required_directories:
        if not directory.exists():
            raise FileNotFoundError(
                "Uma pasta obrigatória não foi encontrada.\n"
                f"Caminho procurado: {directory}"
            )

        number_of_files = len(list(directory.glob("*.txt")))

        print(
            f"{directory.name}: "
            f"{number_of_files} arquivos .txt encontrados"
        )

    print()


def numeric_sort_key(path: Path) -> tuple:
    """
    Permite ordenar arquivos numericamente.

    Exemplo:
    2.txt aparece antes de 10.txt.
    """

    numbers = re.findall(r"\d+", path.stem)

    if numbers:
        return 0, int(numbers[0])

    return 1, path.stem


def parse_publication_date(raw_date: str) -> pd.Timestamp:
    """
    Extrai uma data no formato dia/mês/ano.

    Exemplos aceitos:
    18/10/2017
    20/05/2017 20h52
    20/05/2017 20:52

    O horário é ignorado porque o experimento utiliza
    somente a data de publicação.
    """

    if not raw_date:
        return pd.NaT

    match = re.search(
        r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b",
        raw_date,
    )

    if match is None:
        return pd.NaT

    date_text = match.group(0)

    return pd.to_datetime(
        date_text,
        format="%d/%m/%Y",
        errors="coerce",
    )


def read_file(path: Path) -> str:
    """
    Lê um arquivo de texto usando UTF-8.

    Caracteres inválidos são substituídos para evitar que
    um único arquivo interrompa todo o processamento.
    """

    return path.read_text(
        encoding="utf-8",
        errors="replace",
    ).strip()


def load_class_records(
    class_name: str,
    numeric_label: int,
) -> tuple[list[dict], dict]:
    """
    Carrega os textos e metadados de uma classe.

    class_name:
        "true" ou "fake"

    numeric_label:
        0 para notícias verdadeiras
        1 para notícias falsas
    """

    text_directory = FULL_TEXTS_DIR / class_name
    metadata_directory = (
        FULL_TEXTS_DIR / f"{class_name}-meta-information"
    )

    text_files = sorted(
        text_directory.glob("*.txt"),
        key=numeric_sort_key,
    )

    records = []

    statistics = {
        "texts_found": len(text_files),
        "loaded": 0,
        "missing_metadata": 0,
        "incomplete_metadata": 0,
        "invalid_dates": 0,
        "empty_texts": 0,
    }

    for text_path in text_files:
        document_id = text_path.stem

        metadata_path = (
            metadata_directory / f"{document_id}-meta.txt"
        )

        if not metadata_path.exists():
            statistics["missing_metadata"] += 1

            print(
                "[AVISO] Metadado não encontrado para "
                f"{text_path.name}"
            )
            continue

        try:
            text = read_file(text_path)
            metadata_content = read_file(metadata_path)
        except OSError as error:
            print(
                f"[AVISO] Não foi possível ler {text_path.name}: "
                f"{error}"
            )
            continue

        metadata_lines = metadata_content.splitlines()

        # Estrutura esperada:
        # linha 1: autor
        # linha 2: link
        # linha 3: categoria
        # linha 4: data de publicação
        if len(metadata_lines) < 4:
            statistics["incomplete_metadata"] += 1

            print(
                "[AVISO] Arquivo de metadados incompleto: "
                f"{metadata_path.name}"
            )
            continue

        author = metadata_lines[0].strip()
        url = metadata_lines[1].strip()
        category = metadata_lines[2].strip()
        raw_date = metadata_lines[3].strip()

        publication_date = parse_publication_date(raw_date)

        if pd.isna(publication_date):
            statistics["invalid_dates"] += 1

        if not text:
            statistics["empty_texts"] += 1

        record = {
            "document_id": document_id,
            "label": numeric_label,
            "label_name": class_name,
            "author": author,
            "url": url,
            "category": category,
            "raw_date": raw_date,
            "date": publication_date,
            "text": text,
            "text_path": text_path.relative_to(
                DATASET_DIR
            ).as_posix(),
            "metadata_path": metadata_path.relative_to(
                DATASET_DIR
            ).as_posix(),
        }

        records.append(record)
        statistics["loaded"] += 1

    return records, statistics


def print_loading_statistics(
    class_name: str,
    statistics: dict,
) -> None:
    """
    Exibe as estatísticas do carregamento de uma classe.
    """

    print(f"Classe: {class_name}")
    print(f"  Textos encontrados: {statistics['texts_found']}")
    print(f"  Registros carregados: {statistics['loaded']}")
    print(
        "  Metadados ausentes: "
        f"{statistics['missing_metadata']}"
    )
    print(
        "  Metadados incompletos: "
        f"{statistics['incomplete_metadata']}"
    )
    print(
        "  Datas inválidas: "
        f"{statistics['invalid_dates']}"
    )
    print(
        "  Textos vazios: "
        f"{statistics['empty_texts']}"
    )
    print()


def print_class_distribution(
    dataframe: pd.DataFrame,
    dataset_name: str,
) -> None:
    """
    Exibe a distribuição das classes de um conjunto.
    """

    print(f"Distribuição das classes — {dataset_name}:")

    if dataframe.empty:
        print("  O conjunto está vazio.")
        print()
        return

    counts = dataframe["label_name"].value_counts()

    for class_name, count in counts.items():
        percentage = count / len(dataframe) * 100

        print(
            f"  {class_name}: {count} "
            f"({percentage:.2f}%)"
        )

    print()


def save_dataframe(
    dataframe: pd.DataFrame,
    filename: str,
) -> Path:
    """
    Salva um DataFrame no diretório de saída.
    """

    output_path = OUTPUT_DIR / filename

    dataframe.to_csv(
        output_path,
        index=False,
        encoding="utf-8",
    )

    return output_path


# ============================================================
# EXECUÇÃO PRINCIPAL
# ============================================================

def main() -> None:
    validate_directory_structure()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 70)
    print("CARREGAMENTO DAS NOTÍCIAS")
    print("=" * 70)

    true_records, true_statistics = load_class_records(
        class_name="true",
        numeric_label=0,
    )

    fake_records, fake_statistics = load_class_records(
        class_name="fake",
        numeric_label=1,
    )

    print_loading_statistics(
        "true",
        true_statistics,
    )

    print_loading_statistics(
        "fake",
        fake_statistics,
    )

    all_records = true_records + fake_records

    if not all_records:
        raise RuntimeError(
            "Nenhuma notícia foi encontrada.\n"
            "Verifique se as pastas full_texts/true e "
            "full_texts/fake contêm arquivos .txt."
        )

    dataframe = pd.DataFrame(all_records)

    # Guarda separadamente os registros cuja data não pôde
    # ser interpretada.
    invalid_date_dataframe = dataframe[
        dataframe["date"].isna()
    ].copy()

    # O experimento temporal somente pode usar registros
    # com uma data válida.
    valid_dataframe = dataframe[
        dataframe["date"].notna()
    ].copy()

    if valid_dataframe.empty:
        raise RuntimeError(
            "As notícias foram encontradas, mas nenhuma possui "
            "uma data válida para o corte temporal."
        )

    # Ordenação cronológica.
    # document_id é usado como critério secundário para tornar
    # a ordenação determinística quando duas notícias possuem
    # a mesma data.
    valid_dataframe = (
        valid_dataframe
        .sort_values(
            by=["date", "document_id"],
            kind="stable",
        )
        .reset_index(drop=True)
    )

    train_dataframe = valid_dataframe[
        valid_dataframe["date"] < CUTOFF_DATE
    ].copy()

    test_dataframe = valid_dataframe[
        valid_dataframe["date"] >= CUTOFF_DATE
    ].copy()

    train_dataframe.reset_index(
        drop=True,
        inplace=True,
    )

    test_dataframe.reset_index(
        drop=True,
        inplace=True,
    )

    # Índice cronológico que poderá ser utilizado posteriormente
    # no experimento de aprendizado online.
    test_dataframe["stream_position"] = range(
        1,
        len(test_dataframe) + 1,
    )

    # Salva os arquivos.
    complete_path = save_dataframe(
        valid_dataframe,
        "fakebr_temporal_complete.csv",
    )

    train_path = save_dataframe(
        train_dataframe,
        "train_pre_covid.csv",
    )

    test_path = save_dataframe(
        test_dataframe,
        "test_post_covid.csv",
    )

    invalid_path = None

    if not invalid_date_dataframe.empty:
        invalid_path = save_dataframe(
            invalid_date_dataframe,
            "records_with_invalid_dates.csv",
        )

    print("=" * 70)
    print("RESUMO DO CORTE TEMPORAL")
    print("=" * 70)

    print(
        "Data inicial do corpus: "
        f"{valid_dataframe['date'].min().date()}"
    )

    print(
        "Data final do corpus: "
        f"{valid_dataframe['date'].max().date()}"
    )

    print(
        "Data de corte: "
        f"{CUTOFF_DATE.date()}"
    )

    print()
    print(
        "Treino: notícias anteriores a "
        f"{CUTOFF_DATE.date()}"
    )

    print(
        "Teste: notícias a partir de "
        f"{CUTOFF_DATE.date()}"
    )

    print()
    print(
        f"Total com data válida: {len(valid_dataframe)}"
    )

    print(
        f"Total com data inválida: "
        f"{len(invalid_date_dataframe)}"
    )

    print(
        f"Quantidade no treino: {len(train_dataframe)}"
    )

    print(
        f"Quantidade no teste: {len(test_dataframe)}"
    )

    print()

    print_class_distribution(
        train_dataframe,
        "treino pré-Covid",
    )

    print_class_distribution(
        test_dataframe,
        "teste pós-Covid",
    )

    print("=" * 70)
    print("ARQUIVOS GERADOS")
    print("=" * 70)

    print(complete_path)
    print(train_path)
    print(test_path)

    if invalid_path is not None:
        print(invalid_path)

    if test_dataframe.empty:
        print()
        print("=" * 70)
        print("ATENÇÃO")
        print("=" * 70)
        print(
            "O conjunto de teste ficou vazio. Isso significa "
            "que o Fake.br-Corpus não contém notícias com data "
            "igual ou posterior a 01/12/2019."
        )
        print(
            "Nesse caso, o experimento sobre o período da "
            "Covid-19 não poderá ser realizado somente com "
            "esta base."
        )

    print()
    print("Processamento concluído.")


if __name__ == "__main__":
    main()