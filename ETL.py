import streamlit as st
import pandas as pd
import glob
import re
import os
import numpy as np
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, log_loss
)
import altair as alt
import matplotlib.pyplot as plt

#cargo los archivos obteniendo variables del nombre

def cargar_archivos(patron, columna_nvariables=True, columna_nfolds=True, columna_fold=True):
    archivos = glob.glob(patron)
    dfs = []
    pattern_nv = re.compile(r'nV(\d+)')
    pattern_nf = re.compile(r'nF(\d+)')
    pattern_s = re.compile(r'_S\d+')
    pattern_seed = re.compile(r'Seed[_]?(\d+)')
    pattern_fold = re.compile(r'_Fold(\d+)')

    for archivo in archivos:
        df = pd.read_csv(archivo)
        nombre_archivo = os.path.basename(archivo)

        if columna_nvariables:
            match_nv = pattern_nv.search(archivo)
            if match_nv:
                n_variables = int(match_nv.group(1))
            else:
                n_variables = None
            df['Nvariables'] = n_variables

        if columna_nfolds:
            match_nf = pattern_nf.search(archivo)
            if match_nf:
                n_folds = int(match_nf.group(1))
            else:
                n_folds = None
            df['nFolds'] = n_folds

        if columna_fold:
            match_fold = pattern_fold.search(nombre_archivo)
            if match_fold:
                fold_value = int(match_fold.group(1))
            else:
                fold_value = None
            df['Fold'] = fold_value

        if nombre_archivo.startswith('TestPredCV'):
            df.columns = [pattern_s.sub('', col) for col in df.columns]

        match_seed = pattern_seed.search(nombre_archivo)
        if 'Seed' in nombre_archivo and match_seed:
            seed_value = int(match_seed.group(1))
            df['Seed'] = seed_value

        dfs.append(df)

    df_total = pd.concat(dfs, ignore_index=True)
    return df_total

patron_testpredcv = "DATOS/TestPredCV_ID_2C_NvsSD_nR0_nV*_nF*_Seed*.csv"
df_testpredcv_prev = cargar_archivos(patron_testpredcv, columna_nvariables=True,columna_nfolds=True)

patron_leaderboard = "DATOS/leaderboard_testset_ID_2C_NvsSD_nR0_nV*_nF*_Seed*_Fold*.csv"
df_leaderboard = cargar_archivos(patron_leaderboard, columna_nvariables=True,columna_nfolds=True)

patron_feature_importance = "DATOS/FeatureImportance_ID_2C_NvsSD_nR0_nV*_nF*.csv"
df_feature_importance = cargar_archivos(patron_feature_importance, columna_nvariables=True,columna_nfolds=True)

patron_metrics = "DATOS/Metrics_CV_ID_2C_NvsSD_nR0_nV*_nF*_Seed*.csv"
df_metrics = cargar_archivos(patron_metrics, columna_nvariables=True,columna_nfolds=True)
#extraemos modelos y pivoltamos tabla de muestras para obtener las dimensiones
def extraer_modelos(df):
    modelos = set()
    for col in df.columns:
        match_numfold = re.match(r'testNumFold_(\w+)', col)
        match_proba = re.match(r'testPredProba_(\w+)', col)
        if match_numfold:
            modelos.add(match_numfold.group(1))
        elif match_proba:
            modelos.add(match_proba.group(1))
    return sorted(modelos)

modelos = extraer_modelos(df_testpredcv_prev)
filas_transformadas = []
for _, fila in df_testpredcv_prev.iterrows():
    base_info = {
        'Nvariables': fila['Nvariables'],
        'nFolds': fila['nFolds'],
        'Seed': fila['Seed'],
        'etiq-id': fila['etiq-id'],
        'clasereal': fila['ED_2Clases']
    }
    for modelo in modelos:
        fila_modelo = base_info.copy()
        fila_modelo['model'] = modelo
        test_numfold_col = f'testNumFold_{modelo}'
        test_proba_col = f'testPredProba_{modelo}'
        fila_modelo['testNumFold'] = fila[test_numfold_col] if test_numfold_col in df_testpredcv_prev.columns else None
        fila_modelo['testPredProba'] = fila[test_proba_col] if test_proba_col in df_testpredcv_prev.columns else None
        filas_transformadas.append(fila_modelo)
df_testpredcv = pd.DataFrame(filas_transformadas)
#reordenamos columnas
def reordenar_columnas(df):
    df.columns = [
        'Seed' if col.lower() == 'seed' else col
        for col in df.columns
    ]
    df.columns = [
        'model' if re.search(r'modelname', col, re.IGNORECASE) else col
        for col in df.columns
    ]
    df.columns = [
        'Model' if col.lower() == 'model' else col
        for col in df.columns
    ]
    columnas_prioridad = ['Model', 'Nvariables', 'nFolds', 'Seed', 'Fold']
    columnas_actuales = list(df.columns)
    columnas_prioridad_existentes = [col for col in columnas_prioridad if col in columnas_actuales]
    nuevas_columnas = columnas_prioridad_existentes + [col for col in columnas_actuales if col not in columnas_prioridad_existentes]
    return df[nuevas_columnas]

df_testpredcv = reordenar_columnas(df_testpredcv)
df_leaderboard = reordenar_columnas(df_leaderboard)
df_feature_importance = reordenar_columnas(df_feature_importance)
df_metrics = reordenar_columnas(df_metrics)
#nombres comunes
mapeo_metricas = {
    'precision_macro': 'Precision',
    'Precision_macro': 'Precision_macro',
    'precision': 'Precision',
    'Precision_weighted': 'Precision_weighted',
    'average_precision': 'average_precision',
    'Precision_0': 'Precision_0',
    'Precision_1': 'Precision_1',
    'recall_macro': 'Recall_macro',
    'Recall_macro': 'Recall_macro',
    'recall': 'Recall',
    'Recall_weighted': 'Recall_Weighted',
    'Recall_0': 'Recall_0',
    'Recall_1': 'Recall_1',
    'f1': 'F1',
    'F1_macro': 'F1_macro',
    'f1_micro': 'F1_micro',
    'F1_weighted': 'F1_weighted',
    'balanced_accuracy': 'Balanced_Accuracy',
    'Balanced_accuracy': 'Balanced_Accuracy',
    'roc_auc': 'ROC_AUC',
    'Roc_auc': 'ROC_AUC',
    'auc': 'ROC_AUC',
    'pr_auc': 'PR_AUC',
    'score_test': 'Score_Test',
    'score_val': 'Score_Validation',
    'log_loss': 'Log_Loss',
}
#renombramos columnas
def renombrar_columnas(df, mapeo):
    columnas_actuales = df.columns
    nuevas_columnas = {}
    for col in columnas_actuales:
        col_lower = col.lower()
        if col in mapeo:
            nuevas_columnas[col] = mapeo[col]
        elif col_lower in [k.lower() for k in mapeo]:
            key = [k for k in mapeo if k.lower() == col_lower][0]
            nuevas_columnas[col] = mapeo[key]
        else:
            nuevas_columnas[col] = col
    df_renombrado = df.rename(columns=nuevas_columnas)
    return df_renombrado

df_metrics = renombrar_columnas(df_metrics, mapeo_metricas)
df_leaderboard = renombrar_columnas(df_leaderboard, mapeo_metricas)

mapping = df_metrics.set_index(['Model', 'Nvariables', 'nFolds', 'Seed'])['Roc_auc_byFold'].to_dict()

def get_roc_auc(row):
    key = (row['Model'], row['Nvariables'], row['nFolds'], row['Seed'])
    return mapping.get(key, None)

df_leaderboard['Roc_auc_byFold'] = df_leaderboard.apply(get_roc_auc, axis=1)
df_leaderboard.to_parquet('df_leaderboard.parquet', index=False)
df_testpredcv.to_parquet('df_testpredcv.parquet', index=False)
df_feature_importance.to_parquet('df_feature_importance.parquet', index=False)
df_metrics.to_parquet('df_metrics.parquet', index=False)
print(" ")
print("df_leaderboard.head()")
print(df_leaderboard.head())
print(" ")
print(" ")
print("df_testpredcv.head()")
print(df_testpredcv.head())
print(" ")
print(" ")
print("df_feature_importance.head()")
print(df_feature_importance.head())
print(" ")
print(" ")
print("df_metrics.head()")
print(df_metrics.head())