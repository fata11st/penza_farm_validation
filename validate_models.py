#!/usr/bin/env python3
"""
Полная валидация моделей прогнозирования урожайности
с Bootstrap и обработкой пропущенных значений
"""

import pandas as pd
import numpy as np
import json
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Загрузка данных
df = pd.read_excel('penza_dataset_2025.xlsx')

# Конфигурации моделей
configs = {
    'grain_crops': ('p_grain_crops', 'Зерновые', 'grid_v3/best_p_grain_crops_v3.json'),
    'sugar_beet': ('p_sugar_beet', 'Сахарная свёкла', 'grid_v3/best_p_sugar_beet_v3.json'),
    'sunflower': ('p_sunflower', 'Подсолнечник', 'grid_v3/best_p_sunflower_v3.json'),
    'potato': ('p_potato', 'Картофель', 'grid_v3/best_p_potato_v3.json'),
    'vegetables': ('p_vegetables', 'Овощи', 'grid_v3/best_p_vegetables_v3.json'),
}


def create_features(df, target_col, features):
    """Создание всех необходимых признаков"""
    df = df.copy()
    
    # time_index
    if 'time_index' in features:
        df['time_index'] = df['Год'] - df['Год'].min()
    
    # Лаги и скользящие средние для целевой переменной
    for lag in [1, 2]:
        lag_col = f'{target_col}_lag{lag}'
        if lag_col in features:
            df[lag_col] = df[target_col].shift(lag)
    
    ma_col = f'{target_col}_ma3'
    if ma_col in features:
        df[ma_col] = df[target_col].rolling(window=3).mean()
    
    # Скользящее среднее для других культур
    for other in ['p_grain_crops', 'p_sugar_beet', 'p_sunflower', 'p_potato', 'p_vegetables']:
        if other == target_col:
            continue
        other_ma = f'{other}_ma3'
        if other_ma in features:
            df[other_ma] = df[other].rolling(window=3).mean()
    
    # Агрегированные климатические признаки
    if 'rf_sum_summer' not in df.columns and 'rf_sum_jun' in df.columns:
        df['rf_sum_summer'] = df['rf_sum_jun'] + df['rf_sum_jul'] + df['rf_sum_aug']
    
    if 'mean_temp_summer' not in df.columns and 'mean_temp_jun' in df.columns:
        df['mean_temp_summer'] = (df['mean_temp_jun'] + df['mean_temp_jul'] + df['mean_temp_aug']) / 3
    
    if 'heat_x_moisture' not in df.columns and 'GTK' in df.columns and 'veget_precip' in df.columns:
        df['heat_x_moisture'] = df['GTK'] * df['veget_precip']
    
    if 'frost_risk' not in df.columns and 'mean_temp_mar' in df.columns:
        df['frost_risk'] = (df['mean_temp_mar'] < 0).astype(int)
    
    return df


def get_feature_value(row, feature, df_history, target_col):
    """Получение значения признака с обработкой отсутствующих данных"""
    if feature in row.index:
        val = row[feature]
        if pd.isna(val):
            return 0.0
        return val
    
    # Для лагов и MA вычисляем из истории
    if feature.endswith('_lag1'):
        base_col = feature.replace('_lag1', '')
        if len(df_history) >= 1:
            val = df_history[base_col].iloc[-1]
            return val if not pd.isna(val) else 0.0
        return 0.0
    
    if feature.endswith('_lag2'):
        base_col = feature.replace('_lag2', '')
        if len(df_history) >= 2:
            val = df_history[base_col].iloc[-2]
            return val if not pd.isna(val) else 0.0
        return 0.0
    
    if feature.endswith('_ma3'):
        base_col = feature.replace('_ma3', '')
        if len(df_history) >= 3:
            val = df_history[base_col].tail(3).mean()
            return val if not pd.isna(val) else 0.0
        elif len(df_history) > 0:
            val = df_history[base_col].mean()
            return val if not pd.isna(val) else 0.0
        return 0.0
    
    return 0.0


def bootstrap_predict(X_train, y_train, X_test, scaler, alpha, weights, n_iterations=500):
    """Bootstrap прогнозирование с климатическим пулом"""
    predictions = []
    n_samples = len(X_train)
    
    for _ in range(n_iterations):
        # Bootstrap выборка
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        X_boot = X_train[indices]
        y_boot = y_train[indices]
        w_boot = weights[indices]
        
        # Масштабирование
        scaler_boot = StandardScaler()
        X_boot_scaled = scaler_boot.fit_transform(X_boot)
        X_test_scaled = scaler_boot.transform(X_test)
        
        # Модель
        model = Ridge(alpha=alpha)
        model.fit(X_boot_scaled, y_boot, sample_weight=w_boot)
        
        y_pred = model.predict(X_test_scaled)[0]
        predictions.append(max(y_pred, 0.05))  # Минимальное значение 0.05
    
    return predictions


def validate_model(target_col, target_name, config_path, df):
    """Валидация одной модели с Bootstrap"""
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    features = config['base'] + config.get('extra', [])
    alpha = config['alpha']
    decay = config.get('decay', 1.0)
    min_train = config.get('min_train', 10)
    
    print(f"\n{'='*60}")
    print(f"Модель: {target_name}")
    print(f"Целевая переменная: {target_col}")
    print(f"Признаки ({len(features)}): {features}")
    print(f"Alpha: {alpha}, Decay: {decay}, Min train: {min_train}")
    print('='*60)
    
    # Подготовка данных
    df_sorted = df.sort_values('Год').reset_index(drop=True)
    df_feat = create_features(df_sorted, target_col, features)
    
    results = []
    bootstrap_results = []
    
    for test_year_idx in range(min_train, len(df_feat)):
        train_df = df_feat.iloc[:test_year_idx].copy()
        test_df = df_feat.iloc[test_year_idx:test_year_idx+1].copy()
        
        # Удаляем строки с NaN в целевой переменной
        train_df = train_df.dropna(subset=[target_col])
        
        # Доступные признаки
        available_features = [f for f in features if f in train_df.columns]
        
        # Удаляем строки с NaN в признаках
        train_df = train_df.dropna(subset=available_features)
        
        if len(train_df) < min_train:
            continue
        
        X_train = train_df[available_features].values.astype(float)
        y_train = train_df[target_col].values.astype(float)
        
        # Веса
        weights = np.array([decay ** (len(train_df) - 1 - i) for i in range(len(train_df))])
        
        # Проверка на NaN в тренировочных данных
        if np.any(np.isnan(X_train)):
            print(f"  Год {test_year_idx}: NaN в тренировочных данных, замена на 0")
            X_train = np.nan_to_num(X_train, nan=0.0)
        
        # Масштабирование
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        # Прогноз для тестового года
        X_test_row = test_df[available_features].iloc[0]
        X_test = np.array([[get_feature_value(X_test_row, f, train_df, target_col) 
                           for f in available_features]])
        
        # Дополнительная проверка на NaN
        if np.any(np.isnan(X_test)):
            X_test = np.nan_to_num(X_test, nan=0.0)
        
        X_test_scaled = scaler.transform(X_test)
        
        # Точечный прогноз
        model = Ridge(alpha=alpha)
        model.fit(X_train_scaled, y_train, sample_weight=weights)
        y_pred = model.predict(X_test_scaled)[0]
        y_true = test_df[target_col].values[0]
        
        if np.isnan(y_true):
            continue
        
        mape = abs(y_true - y_pred) / y_true * 100 if y_true != 0 else 0
        results.append({
            'year': int(test_df['Год'].values[0]),
            'y_true': y_true,
            'y_pred': y_pred,
            'mape': mape
        })
        
        # Bootstrap прогноз (для последних 5 лет)
        if test_year_idx >= len(df_feat) - 5:
            boot_preds = bootstrap_predict(
                X_train_scaled, y_train, X_test_scaled, 
                scaler, alpha, weights, n_iterations=500
            )
            bootstrap_results.append({
                'year': int(test_df['Год'].values[0]),
                'median': np.median(boot_preds),
                'ci_lower': np.percentile(boot_preds, 5),
                'ci_upper': np.percentile(boot_preds, 95),
                'std': np.std(boot_preds)
            })
    
    # Результаты валидации
    if results:
        avg_mape = np.mean([r['mape'] for r in results])
        min_mape = np.min([r['mape'] for r in results])
        max_mape = np.max([r['mape'] for r in results])
        
        print(f"\nРезультаты валидации ({len(results)} лет):")
        print(f"  Средний MAPE: {avg_mape:.2f}%")
        print(f"  Минимальный MAPE: {min_mape:.2f}%")
        print(f"  Максимальный MAPE: {max_mape:.2f}%")
        
        # Заявленное значение из конфига
        if 'mape' in config and config['mape']:
            declared_mape = config['mape']
            diff = abs(avg_mape - declared_mape)
            print(f"  Заявленный MAPE: {declared_mape}%")
            print(f"  Расхождение: {diff:.2f}%")
        
        # Bootstrap результаты
        if bootstrap_results:
            print(f"\nBootstrap прогнозы (500 итераций):")
            for br in bootstrap_results:
                print(f"  {br['year']}: медиана={br['median']:.3f}, "
                      f"90% ДИ=[{br['ci_lower']:.3f}, {br['ci_upper']:.3f}], "
                      f"std={br['std']:.3f}")
        
        return avg_mape
    else:
        print("Нет результатов для отображения")
        return None


def main():
    """Основная функция валидации всех моделей"""
    print("="*70)
    print("ВАЛИДАЦИЯ МОДЕЛЕЙ ПРОГНОЗИРОВАНИЯ УРОЖАЙНОСТИ")
    print("Метод: Expanding Window Validation с Bootstrap")
    print("="*70)
    
    all_mapes = {}
    
    for key, (target_col, target_name, config_path) in configs.items():
        try:
            mape = validate_model(target_col, target_name, config_path, df)
            if mape is not None:
                all_mapes[target_name] = mape
        except Exception as e:
            print(f"\nОшибка при валидации {target_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Сводная таблица
    print("\n" + "="*70)
    print("СВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ")
    print("="*70)
    print(f"{'Культура':<25} {'MAPE (%)':<15} {'Статус':<15}")
    print("-"*55)
    
    for name, mape in all_mapes.items():
        status = "OK" if mape < 20 else "Требует внимания"
        print(f"{name:<25} {mape:>8.2f}%      {status:<15}")
    
    print("="*70)
    print("Валидация завершена успешно!")
    print("="*70)


if __name__ == '__main__':
    main()
