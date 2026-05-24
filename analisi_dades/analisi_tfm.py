"""
Anàlisi de dades del TFM
========================

1. carrega les dades de l'Excel de recollida;
2. neteja i unifica les taules;
3. crea les variables derivades;
4. calcula estadístics descriptius i inferencials;
5. guarda totes les taules i gràfics generats.

Requisits:
    pandas, numpy, scipy, statsmodels, matplotlib, openpyxl
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import chi2_contingency, mannwhitneyu, pearsonr, ttest_ind


# =============================================================================
# CONFIGURACIÓ GENERAL
# =============================================================================

SHEETS = {
    "likert": "I1",
    "resultats": "I2 - Resultats",
    "alumnes": "Alumnes",
    "comparacio": "Alumne - Comparació I1 vs I2",
    "examen": "Alumne - Resultats Examen",
}

EXERCICIS = ["Exercici 1", "Exercici 2", "Exercici 3", "Exercici 4"]
ITEMS_LIKERT = [str(i) for i in range(1, 12)]

VARIABLES_ANALISI = [
    "Nota_Exercicis",
    "Rigor",
    "Taxa_Complecio",
    "comprensio_autoeficacia",
    "motivacio_implicacio",
    "claredat_material",
    "baixa_carrega_cognitiva",
]

VARIABLES_PERCEPCIO = [
    "comprensio_autoeficacia",
    "motivacio_implicacio",
    "claredat_material",
    "baixa_carrega_cognitiva",
]

NOMS_VARIABLES = {
    "Nota_Exercicis": "Resultats intervenció",
    "Rigor": "Rigor formal",
    "Taxa_Complecio": "Taxa de compleció",
    "Nota Mitjana Curs": "Mitjana curs",
    "Nota_Mitjana_Curs": "Mitjana curs",
    "Nota_Examen": "Nota examen",
    "comprensio_autoeficacia": "Comprensió i autoeficàcia",
    "motivacio_implicacio": "Motivació i implicació",
    "claredat_material": "Claredat del material",
    "baixa_carrega_cognitiva": "Baixa càrrega cognitiva",
}


# =============================================================================
# FUNCIONS AUXILIARS GENERALS
# =============================================================================

def ensure_dirs(output_dir: Path) -> dict[str, Path]:
    """Crea les carpetes de sortida i retorna les rutes principals."""
    paths = {
        "base": output_dir,
        "tables": output_dir / "taules_TFM",
        "figures": output_dir / "grafics_TFM",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def to_numeric_comma(series: pd.Series) -> pd.Series:
    """Converteix sèries amb coma decimal a valors numèrics."""
    return pd.to_numeric(
        series.astype(str).str.replace(",", ".", regex=False),
        errors="coerce",
    )


def safe_numeric(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Converteix a numèric només les columnes que existeixen."""
    for col in columns:
        if col in df.columns:
            df[col] = to_numeric_comma(df[col])
    return df


def savefig(fig: plt.Figure, path: Path) -> None:
    """Guarda una figura i la tanca."""
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.close(fig)


def style_axis(ax: plt.Axes, title: str, xlabel: str, ylabel: str, ylim=None) -> None:
    """Aplica un estil coherent als gràfics."""
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def configure_matplotlib() -> None:
    """Defineix l'estil global dels gràfics."""
    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.figsize": (8, 5),
            "figure.dpi": 120,
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "axes.titlepad": 12,
        }
    )


# =============================================================================
# CÀRREGA I PREPARACIÓ DE DADES
# =============================================================================

def load_data(input_file: Path) -> dict[str, pd.DataFrame]:
    """Carrega tots els fulls necessaris del fitxer Excel."""
    if not input_file.exists():
        raise FileNotFoundError(f"No s'ha trobat el fitxer: {input_file}")

    data = {
        "likert": pd.read_excel(input_file, sheet_name=SHEETS["likert"], header=1),
        "resultats": pd.read_excel(input_file, sheet_name=SHEETS["resultats"]),
        "alumnes": pd.read_excel(input_file, sheet_name=SHEETS["alumnes"]),
        "comparacio": pd.read_excel(input_file, sheet_name=SHEETS["comparacio"]),
        "examen": pd.read_excel(input_file, sheet_name=SHEETS["examen"]),
    }
    return data


def prepare_likert(df_likert: pd.DataFrame) -> pd.DataFrame:
    """Neteja el qüestionari Likert i crea dimensions agregades."""
    df_likert = df_likert.copy()
    df_likert = safe_numeric(df_likert, ITEMS_LIKERT)

    # Ítems invertits: valors alts han d'indicar menor càrrega cognitiva.
    df_likert["8_inv"] = 6 - df_likert["8"]
    df_likert["11_inv"] = 6 - df_likert["11"]

    df_likert["comprensio_autoeficacia"] = df_likert[["1", "2", "9"]].mean(axis=1)
    df_likert["motivacio_implicacio"] = df_likert[["3", "4"]].mean(axis=1)
    df_likert["claredat_material"] = df_likert[["5", "6", "7", "10"]].mean(axis=1)
    df_likert["baixa_carrega_cognitiva"] = df_likert[["8_inv", "11_inv"]].mean(axis=1)

    return df_likert


def prepare_results(df_resultats: pd.DataFrame) -> pd.DataFrame:
    """Neteja resultats d'intervenció i calcula taxa de compleció."""
    df_resultats = df_resultats.copy()
    numeric_cols = EXERCICIS + [
        "Rigor",
        "Identificacio Problemes",
        "Nota_4",
        "Nota_Exercicis",
        "Intervencio",
    ]
    df_resultats = safe_numeric(df_resultats, numeric_cols)

    df_resultats["Exercicis_No_Respostos"] = (df_resultats[EXERCICIS] == -1).sum(axis=1)
    df_resultats["Exercicis_Contestats"] = (df_resultats[EXERCICIS] != -1).sum(axis=1)
    df_resultats["Taxa_Complecio"] = (
        df_resultats["Exercicis_Contestats"] / len(EXERCICIS)
    ) * 100

    return df_resultats


def prepare_context_tables(data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Normalitza noms de columnes i formats numèrics de les taules contextuals."""
    data = {k: v.copy() for k, v in data.items()}

    data["comparacio"] = data["comparacio"].rename(columns={"Codi": "Identificador"})
    data["examen"] = data["examen"].rename(columns={"Codi": "Identificador"})

    data["examen"] = safe_numeric(
        data["examen"],
        ["Nota_4", "Nota_Examen", "Nota_Mitjana_Curs", "Diferencia_Nota"],
    )
    data["alumnes"] = safe_numeric(data["alumnes"], ["Nota Mitjana Curs"])

    return data


def merge_data(data: dict[str, pd.DataFrame], require_both_interventions: bool = True) -> pd.DataFrame:
    """Uneix resultats, dades d'alumnat, examen, Likert i preferències."""
    data = prepare_context_tables(data)
    df_likert = prepare_likert(data["likert"])
    df_resultats = prepare_results(data["resultats"])

    df = df_resultats.merge(
        data["alumnes"][["Identificador", "Sexe", "Nota Mitjana Curs"]],
        on="Identificador",
        how="left",
    )

    df = df.merge(
        data["examen"][["Identificador", "Nota_4", "Nota_Examen", "Diferencia_Nota"]],
        on="Identificador",
        how="left",
    )

    likert_cols = [
        "Identificador",
        "Intervencio",
        "comprensio_autoeficacia",
        "motivacio_implicacio",
        "claredat_material",
        "baixa_carrega_cognitiva",
    ]
    df = df.merge(df_likert[likert_cols], on=["Identificador", "Intervencio"], how="left")

    df = df.merge(
        data["comparacio"][["Identificador", "Prefereix I1", "Prefereix I2", "Indiferent"]],
        on="Identificador",
        how="left",
    )

    if require_both_interventions:
        participants_valids = (
            df.dropna(subset=["Intervencio"])
            .groupby("Identificador")["Intervencio"]
            .nunique()
            .loc[lambda s: s >= 2]
            .index
        )
        df = df[df["Identificador"].isin(participants_valids)].copy()

    df["Nota_Mitjana_Curs"] = df["Nota Mitjana Curs"]
    df["nivell_previ"] = pd.qcut(
        df["Nota Mitjana Curs"],
        q=3,
        labels=["baix", "mitja", "alt"],
        duplicates="drop",
    )
    df["nivell_matematic"] = df["nivell_previ"]

    df["nivell_resultat_final"] = pd.cut(
        df["Nota_Exercicis"],
        bins=[-0.1, 4.99, 6.99, 10],
        labels=["baix", "mitja", "alt"],
    )

    df["Preferencia"] = df.apply(obtenir_preferencia, axis=1)

    return df


def obtenir_preferencia(row: pd.Series) -> str | float:
    """Crea una variable única de preferència a partir de tres columnes binàries."""
    if row.get("Prefereix I1") == 1:
        return "I1"
    if row.get("Prefereix I2") == 1:
        return "I2"
    if row.get("Indiferent") == 1:
        return "Indiferent"
    return np.nan


def create_preference_df(df: pd.DataFrame) -> pd.DataFrame:
    """Retorna una taula de preferències amb una sola fila per alumne."""
    return (
        df[["Identificador", "Sexe", "Nota Mitjana Curs", "nivell_previ", "Preferencia"]]
        .drop_duplicates(subset=["Identificador"])
        .dropna(subset=["Preferencia"])
        .copy()
    )


# =============================================================================
# ANÀLISI ESTADÍSTICA
# =============================================================================

def compara_intervencions(df: pd.DataFrame, variable: str) -> pd.DataFrame:
    """Compara una variable entre intervencions amb t-test, Mann-Whitney i Cohen's d."""
    dades = df[["Intervencio", variable]].dropna()
    grups = sorted(dades["Intervencio"].unique())
    if len(grups) != 2:
        raise ValueError(f"S'esperaven dues intervencions per a {variable}, però n'hi ha {len(grups)}")

    g1, g2 = grups
    x1 = dades.loc[dades["Intervencio"] == g1, variable]
    x2 = dades.loc[dades["Intervencio"] == g2, variable]

    t_stat, p_ttest = ttest_ind(x1, x2, equal_var=False, nan_policy="omit")
    u_stat, p_mw = mannwhitneyu(x1, x2, alternative="two-sided")

    pooled_sd = np.sqrt((x1.std(ddof=1) ** 2 + x2.std(ddof=1) ** 2) / 2)
    cohens_d = np.nan if pooled_sd == 0 else (x1.mean() - x2.mean()) / pooled_sd

    return pd.DataFrame(
        {
            "variable": [variable],
            "intervencio_1": [g1],
            "intervencio_2": [g2],
            "n_1": [x1.count()],
            "n_2": [x2.count()],
            "mitjana_1": [x1.mean()],
            "mitjana_2": [x2.mean()],
            "desv_1": [x1.std(ddof=1)],
            "desv_2": [x2.std(ddof=1)],
            "mediana_1": [x1.median()],
            "mediana_2": [x2.median()],
            "t_stat": [t_stat],
            "p_ttest": [p_ttest],
            "u_stat": [u_stat],
            "p_mannwhitney": [p_mw],
            "cohens_d": [cohens_d],
        }
    )


def pearson_correlations(df: pd.DataFrame, variables: list[str], target: str) -> pd.DataFrame:
    """Calcula correlacions de Pearson entre una llista de variables i una variable objectiu."""
    results = []
    for var in variables:
        dades = df[[var, target]].dropna()
        if len(dades) < 3:
            r, p = np.nan, np.nan
        else:
            r, p = pearsonr(dades[var], dades[target])
        results.append({"variable": var, "target": target, "r_pearson": r, "p_value": p, "n": len(dades)})
    return pd.DataFrame(results)


def run_anova(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Executa ANOVA i ANCOVA principals."""
    outputs = {}

    model_h4 = smf.ols("Nota_Exercicis ~ C(Intervencio) * C(Sexe)", data=df).fit()
    outputs["anova_H4_genere"] = sm.stats.anova_lm(model_h4, typ=2)

    model_h5 = smf.ols(
        "Nota_Exercicis ~ C(Intervencio) * C(nivell_matematic)",
        data=df,
    ).fit()
    outputs["anova_H5_nivell"] = sm.stats.anova_lm(model_h5, typ=2)

    model_ancova_h4 = smf.ols(
        "Nota_Exercicis ~ C(Intervencio) + C(Sexe) + Nota_Mitjana_Curs + C(Intervencio):C(Sexe)",
        data=df,
    ).fit()
    outputs["ancova_H4_resultats"] = sm.stats.anova_lm(model_ancova_h4, typ=2)

    model_ancova_examen = smf.ols(
        "Nota_Examen ~ C(Intervencio) + C(Sexe) + Nota_Mitjana_Curs + C(Intervencio):C(Sexe)",
        data=df,
    ).fit()
    outputs["ancova_H4_examen"] = sm.stats.anova_lm(model_ancova_examen, typ=2)

    return outputs


def compute_delta_by_sex(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula canvis I2-I1 en variables perceptives segons sexe."""
    results = []
    for sexe in sorted(df["Sexe"].dropna().unique()):
        df_temp = df[df["Sexe"] == sexe]
        for var in VARIABLES_PERCEPCIO:
            mean_i1 = df_temp.loc[df_temp["Intervencio"] == 1, var].mean()
            mean_i2 = df_temp.loc[df_temp["Intervencio"] == 2, var].mean()
            delta = mean_i2 - mean_i1
            pct = np.nan if mean_i1 == 0 else (delta / mean_i1) * 100
            results.append(
                {
                    "Sexe": sexe,
                    "Variable": var,
                    "Mitjana_I1": mean_i1,
                    "Mitjana_I2": mean_i2,
                    "Delta": delta,
                    "Percentatge_canvi": pct,
                }
            )
    return pd.DataFrame(results)


def run_analysis(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Calcula totes les taules estadístiques."""
    outputs: dict[str, pd.DataFrame] = {}

    outputs["descriptius_generals"] = df[VARIABLES_ANALISI].describe().T
    outputs["descriptius_intervencio"] = df.groupby("Intervencio")[VARIABLES_ANALISI].agg(
        ["count", "mean", "std", "median"]
    )
    outputs["descriptius_sexe"] = df.groupby("Sexe")[VARIABLES_ANALISI].agg(
        ["count", "mean", "std", "median"]
    )
    outputs["descriptius_nivell_previ"] = df.groupby("nivell_previ", observed=True)[VARIABLES_ANALISI].agg(
        ["count", "mean", "std", "median"]
    )
    outputs["descriptius_resultat_final"] = df.groupby(
        "nivell_resultat_final", observed=True
    )[VARIABLES_ANALISI].agg(["count", "mean", "std", "median"])

    outputs["intervencio_sexe"] = df.groupby(["Intervencio", "Sexe"])[VARIABLES_ANALISI].mean()
    outputs["intervencio_nivell"] = df.groupby(["Intervencio", "nivell_previ"], observed=True)[
        VARIABLES_ANALISI
    ].mean()
    outputs["intervencio_resultat"] = df.groupby(
        ["Intervencio", "nivell_resultat_final"], observed=True
    )[VARIABLES_ANALISI].mean()

    # Hipòtesis principals
    outputs["H1_rendiment"] = compara_intervencions(df, "Nota_Exercicis")
    outputs["H2_rigor"] = compara_intervencions(df, "Rigor")
    outputs["H3_comprensio"] = compara_intervencions(df, "comprensio_autoeficacia")
    outputs["H3_motivacio"] = compara_intervencions(df, "motivacio_implicacio")
    outputs["H3_claredat"] = compara_intervencions(df, "claredat_material")
    outputs["H3_carrega"] = compara_intervencions(df, "baixa_carrega_cognitiva")

    outputs.update(run_anova(df))

    outputs["corr_percepcio_resultats"] = pearson_correlations(
        df, VARIABLES_PERCEPCIO, "Nota_Exercicis"
    )
    outputs["corr_percepcio_examen"] = pearson_correlations(df, VARIABLES_PERCEPCIO, "Nota_Examen")

    outputs["matriu_correlacions"] = df[
        ["Nota Mitjana Curs", "Nota_Exercicis", *VARIABLES_PERCEPCIO]
    ].corr()

    df_pref = create_preference_df(df)
    outputs["preferencies_alumne"] = df_pref
    outputs["preferencies_global"] = df_pref["Preferencia"].value_counts().rename_axis("Preferencia").to_frame("n")
    outputs["preferencies_sexe"] = pd.crosstab(df_pref["Sexe"], df_pref["Preferencia"])
    outputs["preferencies_nivell"] = pd.crosstab(df_pref["nivell_previ"], df_pref["Preferencia"])

    outputs["chi2_preferencia_sexe"] = chi2_table(outputs["preferencies_sexe"])
    outputs["chi2_preferencia_nivell"] = chi2_table(outputs["preferencies_nivell"])

    outputs["delta_percepcio_sexe"] = compute_delta_by_sex(df)
    outputs["delta_resultats_sexe"] = compute_delta_results_by_sex(df)

    outputs["descriptiu_mostra_alumne"] = create_student_level_descriptives(df)
    outputs["resum_mitjana_curs"] = pd.DataFrame(
        {
            "Mitjana": [outputs["descriptiu_mostra_alumne"]["Mitjana_curs"].mean()],
            "Variancia": [outputs["descriptiu_mostra_alumne"]["Mitjana_curs"].var()],
            "Desviacio_tipica": [outputs["descriptiu_mostra_alumne"]["Mitjana_curs"].std()],
        }
    )

    return outputs


def chi2_table(contingency: pd.DataFrame) -> pd.DataFrame:
    """Retorna el resultat del test chi-quadrat per una taula de contingència."""
    if contingency.empty or contingency.shape[0] < 2 or contingency.shape[1] < 2:
        return pd.DataFrame({"chi2": [np.nan], "p_value": [np.nan], "dof": [np.nan]})
    chi2, p, dof, _ = chi2_contingency(contingency)
    return pd.DataFrame({"chi2": [chi2], "p_value": [p], "dof": [dof]})


def compute_delta_results_by_sex(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula el canvi I2-I1 en resultats d'intervenció segons sexe."""
    pivot = df.pivot_table(
        index=["Identificador", "Sexe"],
        columns="Intervencio",
        values="Nota_Exercicis",
    ).reset_index()

    if 1 in pivot.columns and 2 in pivot.columns:
        pivot["Delta_Resultats"] = pivot[2] - pivot[1]
    else:
        pivot["Delta_Resultats"] = np.nan

    return pivot.groupby("Sexe")["Delta_Resultats"].mean().rename("Delta_Resultats").to_frame()


def create_student_level_descriptives(df: pd.DataFrame) -> pd.DataFrame:
    """Crea una taula amb una fila per alumne."""
    out = (
        df.groupby(["Identificador", "Sexe", "Nota Mitjana Curs", "Nota_Examen"])[
            ["Nota_Exercicis", *VARIABLES_PERCEPCIO]
        ]
        .mean()
        .reset_index()
    )
    return out.rename(
        columns={
            "Nota Mitjana Curs": "Mitjana_curs",
            "Nota_Examen": "Nota_examen",
            "Nota_Exercicis": "Resultats_intervencio",
            "comprensio_autoeficacia": "Comprensio_autoeficacia",
            "motivacio_implicacio": "Motivacio_implicacio",
            "claredat_material": "Claredat_material",
            "baixa_carrega_cognitiva": "Baixa_carrega_cognitiva",
        }
    )


# =============================================================================
# EXPORTACIÓ DE TAULES
# =============================================================================

def save_tables(df: pd.DataFrame, outputs: dict[str, pd.DataFrame], paths: dict[str, Path]) -> None:
    """Guarda totes les taules en Excel i CSV."""
    excel_path = paths["tables"] / "resultats_analisi_TFM.xlsx"

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="dades_unificades", index=False)
        for name, table in outputs.items():
            sheet = name[:31]
            table.to_excel(writer, sheet_name=sheet)

    # També guardem CSV individuals per facilitar revisió i traçabilitat.
    df.to_csv(paths["tables"] / "dades_unificades.csv", index=False)
    for name, table in outputs.items():
        table.to_csv(paths["tables"] / f"{name}.csv")

    # Fitxer específic utilitzat al text descriptiu.
    outputs["descriptiu_mostra_alumne"].to_excel(
        paths["tables"] / "taula_descriptiva_mostra.xlsx",
        index=False,
    )


# =============================================================================
# GRÀFICS
# =============================================================================

def plot_basic_outputs(df: pd.DataFrame, outputs: dict[str, pd.DataFrame], fig_dir: Path) -> None:
    """Genera i guarda tots els gràfics emprats en l'anàlisi."""
    # 01 Resultats intervenció per intervenció
    fig, ax = plt.subplots(figsize=(7, 5))
    df.boxplot(column="Nota_Exercicis", by="Intervencio", ax=ax)
    plt.suptitle("")
    style_axis(ax, "Resultats segons la intervenció", "Intervenció", "Resultats (0-10)", (0, 10))
    savefig(fig, fig_dir / "01_resultats_intervencio_per_intervencio.png")

    # 02 Rigor formal
    fig, ax = plt.subplots(figsize=(7, 5))
    df.boxplot(column="Rigor", by="Intervencio", ax=ax)
    plt.suptitle("")
    style_axis(ax, "Rigor formal segons la intervenció", "Intervenció", "Rigor formal (1-5)", (1, 5))
    savefig(fig, fig_dir / "02_rigor_formal_per_intervencio.png")

    # 03 Taxa compleció
    fig, ax = plt.subplots(figsize=(7, 5))
    df.boxplot(column="Taxa_Complecio", by="Intervencio", ax=ax)
    plt.suptitle("")
    style_axis(ax, "Taxa de compleció segons la intervenció", "Intervenció", "Taxa de compleció (%)", (0, 100))
    savefig(fig, fig_dir / "03_taxa_complecio_per_intervencio.png")

    # 04 Dimensions Likert
    means = df.groupby("Intervencio")[VARIABLES_PERCEPCIO].mean().T
    means.index = [NOMS_VARIABLES[v] for v in means.index]
    fig, ax = plt.subplots(figsize=(10, 6))
    means.plot(kind="bar", ax=ax)
    style_axis(ax, "Dimensions perceptives segons la intervenció", "Variable", "Mitjana Likert (1-5)", (1, 5))
    ax.tick_params(axis="x", rotation=35)
    ax.legend(title="Intervenció")
    savefig(fig, fig_dir / "04_dimensions_likert_per_intervencio.png")

    # 05 Resultats per sexe
    barplot_series(
        df.groupby("Sexe")["Nota_Exercicis"].mean(),
        fig_dir / "05_resultats_intervencio_per_sexe.png",
        "Resultats segons el sexe",
        "Sexe",
        "Resultats (0-10)",
        ylim=(0, 10),
    )

    # 06 Resultats per nivell previ
    barplot_series(
        df.groupby("nivell_previ", observed=True)["Nota_Exercicis"].mean(),
        fig_dir / "06_resultats_intervencio_per_mitjana_curs.png",
        "Resultats segons la mitjana del curs",
        "Nivell segons mitjana del curs",
        "Resultats (0-10)",
        ylim=(0, 10),
    )

    # 07 Resultats segons sexe i intervenció
    grouped_barplot(
        df.groupby(["Sexe", "Intervencio"])["Nota_Exercicis"].mean().unstack(),
        fig_dir / "07_resultats_intervencio_sexe_intervencio.png",
        "Resultats segons sexe i intervenció",
        "Sexe",
        "Resultats (0-10)",
        ylim=(0, 10),
    )

    # 08 Resultats segons nivell i intervenció
    grouped_barplot(
        df.groupby(["nivell_previ", "Intervencio"], observed=True)["Nota_Exercicis"].mean().unstack(),
        fig_dir / "08_resultats_intervencio_mitjana_curs_intervencio.png",
        "Resultats segons mitjana del curs i intervenció",
        "Nivell segons mitjana del curs",
        "Resultats (0-10)",
        ylim=(0, 10),
    )

    # 09 Claredat segons nivell i intervenció
    grouped_barplot(
        df.groupby(["nivell_previ", "Intervencio"], observed=True)["claredat_material"].mean().unstack(),
        fig_dir / "09_claredat_material_mitjana_curs_intervencio.png",
        "Claredat segons mitjana del curs i intervenció",
        "Nivell segons mitjana del curs",
        "Claredat (Likert 1-5)",
        ylim=(1, 5),
    )

    # 10 Baixa càrrega cognitiva segons nivell i intervenció
    grouped_barplot(
        df.groupby(["nivell_previ", "Intervencio"], observed=True)["baixa_carrega_cognitiva"].mean().unstack(),
        fig_dir / "10_baixa_carrega_mitjana_curs_intervencio.png",
        "Baixa càrrega cognitiva segons mitjana del curs i intervenció",
        "Nivell segons mitjana del curs",
        "Baixa càrrega cognitiva (Likert 1-5)",
        ylim=(1, 5),
    )

    # 11 / alias 13: relació mitjana curs-resultats
    scatterplot(
        df,
        "Nota Mitjana Curs",
        "Nota_Exercicis",
        fig_dir / "11_relacio_mitjana_curs_resultats_intervencio.png",
        "Relació entre mitjana del curs i resultats",
        "Mitjana curs",
        "Resultats intervenció (0-10)",
        xlim=(0, 10),
        ylim=(0, 10),
    )
    scatterplot(
        df,
        "Nota Mitjana Curs",
        "Nota_Exercicis",
        fig_dir / "13_nota_previa_rendiment.png",
        "Relació entre nivell previ i rendiment",
        "Mitjana curs",
        "Resultats intervenció (0-10)",
        xlim=(0, 10),
        ylim=(0, 10),
    )

    # 12 relació examen-resultats
    scatterplot(
        df,
        "Nota_Examen",
        "Nota_Exercicis",
        fig_dir / "12_relacio_nota_examen_resultats_intervencio.png",
        "Relació entre nota d'examen i resultats",
        "Nota examen",
        "Resultats intervenció (0-10)",
        xlim=(0, 10),
        ylim=(0, 10),
    )

    # Heatmaps
    heatmap_correlations(
        df,
        ["Nota_Examen", *VARIABLES_PERCEPCIO],
        fig_dir / "13_heatmap_correlacions_percepcio_nota_examen.png",
        "Correlacions entre percepció i nota d'examen",
    )
    heatmap_correlations(
        df,
        ["Nota Mitjana Curs", "Nota_Exercicis", *VARIABLES_PERCEPCIO],
        fig_dir / "correlacions_mitjana_altres.png",
        "Correlacions entre nivell previ, rendiment i variables perceptives",
    )

    # Scatterplots percepció vs nota examen
    for var in VARIABLES_PERCEPCIO:
        scatterplot(
            df,
            var,
            "Nota_Examen",
            fig_dir / f"14_scatter_{var}_nota_examen.png",
            f"{NOMS_VARIABLES[var]} i nota d'examen",
            f"{NOMS_VARIABLES[var]} (Likert 1-5)",
            "Nota examen",
            xlim=(1, 5),
            ylim=(0, 10),
        )

    # Preferències
    plot_preferences(outputs, fig_dir)

    # Interacció gènere/intervenció i deltes
    plot_gender_interaction(df, fig_dir)
    plot_delta_by_sex(outputs["delta_percepcio_sexe"], fig_dir)
    plot_delta_results_by_sex(outputs["delta_resultats_sexe"], fig_dir)


def barplot_series(series: pd.Series, path: Path, title: str, xlabel: str, ylabel: str, ylim=None) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    series.plot(kind="bar", ax=ax)
    style_axis(ax, title, xlabel, ylabel, ylim)
    ax.tick_params(axis="x", rotation=0)
    savefig(fig, path)


def grouped_barplot(table: pd.DataFrame, path: Path, title: str, xlabel: str, ylabel: str, ylim=None) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    table.plot(kind="bar", ax=ax)
    style_axis(ax, title, xlabel, ylabel, ylim)
    ax.tick_params(axis="x", rotation=0)
    ax.legend(title="Intervenció")
    savefig(fig, path)


def scatterplot(
    df: pd.DataFrame,
    x: str,
    y: str,
    path: Path,
    title: str,
    xlabel: str,
    ylabel: str,
    xlim=None,
    ylim=None,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(df[x], df[y], alpha=0.7)
    style_axis(ax, title, xlabel, ylabel, ylim)
    if xlim is not None:
        ax.set_xlim(xlim)
    savefig(fig, path)


def heatmap_correlations(df: pd.DataFrame, variables: list[str], path: Path, title: str) -> None:
    corr = df[variables].corr()
    labels = [NOMS_VARIABLES.get(v, v) for v in variables]

    fig, ax = plt.subplots(figsize=(8, 7))
    cax = ax.imshow(corr, cmap="Blues", vmin=-1, vmax=1)

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", color="black")

    ax.set_title(title, fontweight="bold")
    fig.colorbar(cax, ax=ax, label="Coeficient de correlació")
    savefig(fig, path)


def plot_preferences(outputs: dict[str, pd.DataFrame], fig_dir: Path) -> None:
    pref_global = outputs["preferencies_global"]["n"]
    barplot_series(
        pref_global,
        fig_dir / "preferencia_global.png",
        "Preferència global de material",
        "Preferència",
        "Nombre d'alumnes",
    )

    grouped_barplot(
        outputs["preferencies_sexe"],
        fig_dir / "preferencia_GENERE.png",
        "Preferència de material segons sexe",
        "Sexe",
        "Nombre d'alumnes",
    )

    grouped_barplot(
        outputs["preferencies_nivell"],
        fig_dir / "preferencia-nivell.png",
        "Preferència de material segons mitjana del curs",
        "Nivell segons mitjana del curs",
        "Nombre d'alumnes",
    )


def plot_gender_interaction(df: pd.DataFrame, fig_dir: Path) -> None:
    """Gràfic de motivació i resultats segons intervenció i sexe."""
    motivacio = df.groupby(["Intervencio", "Sexe"])["motivacio_implicacio"].mean().unstack()
    resultats = df.groupby(["Intervencio", "Sexe"])["Nota_Exercicis"].mean().unstack()

    sexes = [s for s in ["H", "D"] if s in motivacio.columns]
    x = np.arange(len(motivacio.index))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(8, 5))

    if "H" in sexes:
        ax1.bar(x - width / 2, motivacio["H"], width, label="Homes - motivació", alpha=0.8)
    if "D" in sexes:
        ax1.bar(x + width / 2, motivacio["D"], width, label="Dones - motivació", alpha=0.8)

    ax1.set_ylabel("Motivació / implicació")
    ax1.set_ylim(1, 5)
    ax1.set_xlabel("Intervenció")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"I{int(i)}" for i in motivacio.index])
    ax1.set_title("Motivació i resultats segons intervenció i sexe", fontweight="bold")

    ax2 = ax1.twinx()
    if "H" in sexes:
        ax2.plot(x, resultats["H"], marker="o", linewidth=2, label="Homes - resultats")
    if "D" in sexes:
        ax2.plot(x, resultats["D"], marker="o", linewidth=2, label="Dones - resultats")
    ax2.set_ylabel("Resultats intervenció")
    ax2.set_ylim(0, 10)

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    fig.legend(handles1 + handles2, labels1 + labels2, loc="center left", bbox_to_anchor=(1.02, 0.5))

    savefig(fig, fig_dir / "interaccio_genere_intervencio.png")


def plot_delta_by_sex(df_delta: pd.DataFrame, fig_dir: Path) -> None:
    pivot = df_delta.pivot(index="Variable", columns="Sexe", values="Delta")
    labels = ["Comprensió", "Motivació", "Claredat", "Baixa càrrega"]

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(pivot.index))
    width = 0.35

    if "H" in pivot.columns:
        ax.bar(x - width / 2, pivot["H"], width, label="Homes")
    if "D" in pivot.columns:
        ax.bar(x + width / 2, pivot["D"], width, label="Dones")

    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels[: len(pivot.index)])
    style_axis(ax, "Canvi entre intervencions segons sexe", "Variable", "Increment I2 - I1", (-2, 2))
    ax.legend(title="Sexe")

    savefig(fig, fig_dir / "increment_motivacio_intervencio.png")
    # Àlies amb accent, per compatibilitat amb el nom utilitzat al document LaTeX.
    fig, ax = plt.subplots(figsize=(9, 5))
    if "H" in pivot.columns:
        ax.bar(x - width / 2, pivot["H"], width, label="Homes")
    if "D" in pivot.columns:
        ax.bar(x + width / 2, pivot["D"], width, label="Dones")
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels[: len(pivot.index)])
    style_axis(ax, "Canvi entre intervencions segons sexe", "Variable", "Increment I2 - I1", (-2, 2))
    ax.legend(title="Sexe")
    savefig(fig, fig_dir / "increment_motivació_intervencio.png")


def plot_delta_results_by_sex(delta_results: pd.DataFrame, fig_dir: Path) -> None:
    barplot_series(
        delta_results["Delta_Resultats"],
        fig_dir / "delta_resultats_sexe.png",
        "Increment de resultats entre intervencions",
        "Sexe",
        "Δ resultats (I2 - I1)",
        ylim=(-5, 5),
    )


# =============================================================================
# PROGRAMA PRINCIPAL
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Anàlisi de dades del TFM")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("Dades/Recollida.xlsx"),
        help="Ruta del fitxer Excel d'entrada.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("Codis anàlisi"),
        help="Carpeta on es guardaran taules i gràfics.",
    )
    parser.add_argument(
        "--keep-incomplete",
        action="store_true",
        help="No descarta alumnes que només han participat en una intervenció.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = ensure_dirs(args.output)
    configure_matplotlib()

    print("Carregant dades...")
    data = load_data(args.input)

    print("Preparant i unint dades...")
    df = merge_data(data, require_both_interventions=not args.keep_incomplete)

    print("Executant anàlisi...")
    outputs = run_analysis(df)

    print("Guardant taules...")
    save_tables(df, outputs, paths)

    print("Generant gràfics...")
    plot_basic_outputs(df, outputs, paths["figures"])

    print("\nAnàlisi completada correctament.")
    print(f"Dades unificades: {paths['tables'] / 'dades_unificades.csv'}")
    print(f"Resultats Excel:  {paths['tables'] / 'resultats_analisi_TFM.xlsx'}")
    print(f"Gràfics:          {paths['figures']}")
    print("\nMostra final:")
    print(f"  Files base de dades: {len(df)}")
    print(f"  Alumnes únics:       {df['Identificador'].nunique()}")


if __name__ == "__main__":
    main()
