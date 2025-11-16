import 'dart:convert';
import 'dart:math';
import 'package:flutter/services.dart';
import 'package:mangoat/models/product_data.dart';
import 'package:mangoat/services/preprocessing_service.dart';

/// Servicio de predicción de demanda basado en el modelo entrenado
class PredictionService {
  // Parámetros del modelo cargados desde el JSON
  Map<String, dynamic>? _modelParams;
  bool _isInitialized = false;

  // Estadísticas para normalización
  late Map<String, double> _numericMeans;
  late Map<String, double> _numericScales;

  // Dataset para KNN
  final List<_FeatureSpec> _featureSchema = [];
  final List<_FeatureStat> _featureStats = [];
  final List<List<double>> _knnSamples = [];
  final List<double> _knnTargets = [];
  int _defaultK = 25;
  bool _knnLoaded = false;

  /// Inicializa el servicio cargando los parámetros del modelo
  Future<void> initialize() async {
    if (_isInitialized) return;

    try {
      // Cargar parámetros del modelo desde assets
      final String jsonString = await rootBundle.loadString(
        'assets/model_params.json',
      );
      _modelParams = json.decode(jsonString);

      // Extraer estadísticas de normalización
      final numericPreprocessing = _modelParams!['numeric_preprocessing'];
      final List<dynamic> means = numericPreprocessing['scaler_mean'];
      final List<dynamic> scales = numericPreprocessing['scaler_scale'];
      final List<dynamic> features = numericPreprocessing['features'];

      _numericMeans = {};
      _numericScales = {};

      for (int i = 0; i < features.length; i++) {
        _numericMeans[features[i]] = means[i].toDouble();
        _numericScales[features[i]] = scales[i].toDouble();
      }

      await _loadKnnDataset();

      _isInitialized = true;
      print('✓ Servicio de predicción inicializado');
    } catch (e) {
      print('Error al inicializar el servicio de predicción: $e');
      // Si falla, usamos valores por defecto
      _initializeDefaults();
    }
  }

  /// Inicializa con valores por defecto si no se pueden cargar los parámetros
  void _initializeDefaults() {
    _numericMeans = {
      'life_cycle_length': 12.0,
      'num_stores': 150.0,
      'num_sizes': 5.0,
      'has_plus_sizes': 0.3,
      'price': 30.0,
      'month_sin': 0.0,
      'month_cos': 1.0,
      'similarity_to_season_center': 0.5,
      'avg_distance_within_season': 1.0,
      'similarity_to_nearest_seasons': 0.5,
    };

    _numericScales = {
      'life_cycle_length': 5.0,
      'num_stores': 100.0,
      'num_sizes': 2.0,
      'has_plus_sizes': 0.5,
      'price': 20.0,
      'month_sin': 0.7,
      'month_cos': 0.7,
      'similarity_to_season_center': 0.3,
      'avg_distance_within_season': 0.5,
      'similarity_to_nearest_seasons': 0.3,
    };

    _isInitialized = true;
    print('⚠ Servicio inicializado con valores por defecto');
  }

  Future<void> _loadKnnDataset() async {
    if (_knnLoaded) return;
    try {
      final jsonString = await rootBundle.loadString('assets/knn_dataset.json');
      final Map<String, dynamic> data = json.decode(jsonString);

      _featureSchema
        ..clear()
        ..addAll(
          (data['schema'] as List<dynamic>).map((item) {
            final map = item as Map<String, dynamic>;
            final type = (map['type'] as String?)?.toLowerCase() ?? 'numeric';
            return _FeatureSpec(
              name: map['name'] as String,
              isNumeric: type == 'numeric',
            );
          }),
        );

      _featureStats
        ..clear()
        ..addAll(
          (data['feature_stats'] as List<dynamic>).map((item) {
            final map = item as Map<String, dynamic>;
            return _FeatureStat(
              mean: (map['mean'] as num).toDouble(),
              std: (map['std'] as num).toDouble(),
            );
          }),
        );

      _knnSamples.clear();
      _knnTargets.clear();

      final samples = data['samples'] as List<dynamic>;
      for (final sample in samples) {
        final map = sample as Map<String, dynamic>;
        final features = (map['features'] as List<dynamic>)
            .map((value) => (value as num).toDouble())
            .toList();
        final target = (map['target'] as num).toDouble();

        if (_featureSchema.isNotEmpty &&
            features.length == _featureSchema.length) {
          _knnSamples.add(features);
          _knnTargets.add(target);
        }
      }

      _defaultK = (data['k'] as num?)?.toInt() ?? 25;
      _knnLoaded = _knnSamples.isNotEmpty;
      if (_knnLoaded) {
        print('✓ Dataset KNN cargado (${_knnSamples.length} muestras)');
      } else {
        print('⚠ Dataset KNN vacío, se usará el modelo heurístico');
      }
    } catch (e) {
      _knnLoaded = false;
      print('Error al cargar dataset KNN: $e');
    }
  }

  List<double>? _buildStandardizedVector(Map<String, dynamic> features) {
    if (_featureSchema.isEmpty ||
        _featureSchema.length != _featureStats.length) {
      return null;
    }

    final vector = <double>[];
    for (int i = 0; i < _featureSchema.length; i++) {
      final spec = _featureSchema[i];
      final stat = _featureStats[i];
      final rawValue = spec.isNumeric
          ? _getNumericFeature(features, spec.name)
          : _stableStringToUnit(features[spec.name]?.toString());
      final standardized = (rawValue - stat.mean) / stat.safeStd;
      vector.add(standardized);
    }

    return vector;
  }

  double _getNumericFeature(Map<String, dynamic> features, String name) {
    final value = features[name];
    if (value is num) return value.toDouble();
    if (value is String) return double.tryParse(value) ?? 0.0;
    return 0.0;
  }

  double _stableStringToUnit(String? value) {
    const modulus = 1000003;
    const multiplier = 131;
    final text = value ?? '';
    int hash = 0;
    for (final codeUnit in text.codeUnits) {
      hash = (hash * multiplier + codeUnit) % modulus;
    }
    return hash / modulus;
  }

  double? _predictWithKnn(Map<String, dynamic> features, {int? k}) {
    if (!_knnLoaded || _knnSamples.isEmpty) return null;
    final vector = _buildStandardizedVector(features);
    if (vector == null) return null;

    final neighborCount = min(k ?? _defaultK, _knnSamples.length);
    if (neighborCount == 0) return null;

    final neighbors = <_Neighbor>[];
    for (int i = 0; i < _knnSamples.length; i++) {
      final sample = _knnSamples[i];
      if (sample.length != vector.length) continue;
      final distance = PreprocessingService.euclideanDistance(vector, sample);
      neighbors.add(_Neighbor(distance: distance, target: _knnTargets[i]));
    }

    if (neighbors.isEmpty) return null;

    neighbors.sort((a, b) => a.distance.compareTo(b.distance));
    final selected = neighbors.take(neighborCount);

    double weightedSum = 0.0;
    double totalWeight = 0.0;
    for (final neighbor in selected) {
      final weight = 1.0 / (neighbor.distance + 1e-6);
      weightedSum += weight * neighbor.target;
      totalWeight += weight;
    }

    if (totalWeight == 0.0) return null;
    return weightedSum / totalWeight;
  }

  /// Predice la demanda semanal para un producto
  ///
  /// Este es un modelo simplificado basado en heurísticas que aproxima
  /// el comportamiento del modelo LightGBM entrenado en Python.
  ///
  /// Para una implementación completa, se necesitaría:
  /// 1. Un intérprete de árboles de decisión en Dart
  /// 2. Los 1500 árboles del modelo exportados en formato JSON
  /// 3. Una biblioteca de ML específica para Flutter
  Future<double> predictDemand(ProductData product) async {
    if (!_isInitialized) {
      await initialize();
    }

    // 1. Preprocesar el producto
    final features = PreprocessingService.processProduct(product);

    // 2. Extraer features numéricas
    final numericFeatures = <String, double>{
      'life_cycle_length': features['life_cycle_length'],
      'num_stores': features['num_stores'],
      'num_sizes': features['num_sizes'],
      'has_plus_sizes': features['has_plus_sizes'],
      'price': features['price'],
      'month_sin': features['month_sin'],
      'month_cos': features['month_cos'],
      'similarity_to_season_center': features['similarity_to_season_center'],
      'avg_distance_within_season': features['avg_distance_within_season'],
      'similarity_to_nearest_seasons':
          features['similarity_to_nearest_seasons'],
    };

    // 3. Normalizar features numéricas
    final normalized = PreprocessingService.normalizeNumericFeatures(
      numericFeatures,
      _numericMeans,
      _numericScales,
    );

    // 4. Aplicar modelo heurístico (aproximación del modelo real)
    double prediction = _applyHeuristicModel(normalized, features);

    final knnPrediction = _predictWithKnn(features);
    if (knnPrediction != null) {
      prediction = _blendPredictions(knnPrediction, prediction);
    }

    // 5. Asegurar que no sea negativo
    return max(0.0, prediction);
  }

  /// Modelo heurístico que aproxima el comportamiento del LightGBM entrenado
  ///
  /// Este modelo usa reglas basadas en las estadísticas del modelo real:
  /// - Media de predicciones: 11908.60
  /// - Mediana: 7262.63
  /// - Rango: 21.81 - 180242.71
  double _applyHeuristicModel(
    Map<String, double> normalized,
    Map<String, dynamic> features,
  ) {
    // Base de demanda (mediana del modelo real)
    double baseDemand = 7262.63;

    // Factor de ciclo de vida (productos con más ciclo tienden a vender más)
    final lifeCycleFactor = 1.0 + (normalized['life_cycle_length']! * 0.3);

    // Factor de distribución (más tiendas = más demanda)
    final storesFactor = 1.0 + (normalized['num_stores']! * 0.5);

    // Factor de tallas (más variedad = más potencial)
    final sizesFactor = 1.0 + (normalized['num_sizes']! * 0.2);

    // Factor de precio (relación inversa, precios muy altos reducen demanda)
    final priceFactor = 1.0 - (normalized['price']! * 0.1).clamp(-0.5, 0.5);

    // Factor estacional (Fourier features)
    final seasonalFactor =
        1.0 +
        (normalized['month_sin']! * 0.2) +
        (normalized['month_cos']! * 0.1);

    // Factor de similitud estacional (productos típicos de la temporada venden más)
    final similarityFactor =
        1.0 + (normalized['similarity_to_season_center']! * 0.3);

    // Factores categóricos (basados en patrones comunes)
    double categoryFactor = 1.0;

    // Factor de familia de producto
    final family = features['aggregated_family'] as String;
    if (family == 'Woman') {
      categoryFactor *= 1.2; // Las prendas de mujer suelen tener mayor demanda
    } else if (family == 'Man') {
      categoryFactor *= 0.9;
    }

    // Factor de categoría
    final category = features['category'] as String;
    if (category.toLowerCase().contains('dress') ||
        category.toLowerCase().contains('shirt') ||
        category.toLowerCase().contains('pant')) {
      categoryFactor *= 1.15; // Categorías populares
    }

    // Factor de estación (tendencias estacionales)
    final season = features['season'] as String;
    final sleeveType = features['sleeve_length_type'] as String;

    // Manga larga en invierno = más demanda
    if (season == 'Winter' && sleeveType.toLowerCase().contains('long')) {
      categoryFactor *= 1.25;
    }
    // Manga corta en verano = más demanda
    else if (season == 'Summer' && sleeveType.toLowerCase().contains('short')) {
      categoryFactor *= 1.25;
    }

    // Factor de tejido
    final fabric = features['fabric'] as String;
    if (fabric.toLowerCase().contains('cotton') ||
        fabric.toLowerCase().contains('linen')) {
      categoryFactor *= 1.1; // Tejidos populares
    }

    // Calcular demanda final
    double demand =
        baseDemand *
        lifeCycleFactor *
        storesFactor *
        sizesFactor *
        priceFactor *
        seasonalFactor *
        similarityFactor *
        categoryFactor;

    // Añadir variabilidad aleatoria (±15%)
    final random = Random();
    final randomFactor = 0.85 + (random.nextDouble() * 0.3);
    demand *= randomFactor;

    // Limitar al rango observado en el modelo real
    return demand.clamp(21.81, 180242.71);
  }

  double _blendPredictions(double knnPrediction, double heuristicPrediction) {
    const knnWeight = 0.7;
    const heuristicWeight = 0.3;
    return (knnPrediction * knnWeight) +
        (heuristicPrediction * heuristicWeight);
  }

  /// Predice demanda y devuelve información detallada
  Future<PredictionResult> predictWithDetails(ProductData product) async {
    final demand = await predictDemand(product);
    final features = PreprocessingService.processProduct(product);

    return PredictionResult(
      predictedDemand: demand,
      season: features['season'] as String,
      confidence: _calculateConfidence(features),
      factors: _extractKeyFactors(features, demand),
    );
  }

  /// Calcula un nivel de confianza para la predicción (0-100)
  double _calculateConfidence(Map<String, dynamic> features) {
    double confidence = 70.0; // Base

    // Aumentar confianza si hay buenas características
    if (features['num_stores'] > 100) confidence += 5;
    if (features['num_sizes'] > 3) confidence += 5;
    if (features['similarity_to_season_center'] > 0.6) confidence += 10;
    if (features['price'] > 15 && features['price'] < 60) confidence += 5;

    return confidence.clamp(0, 100);
  }

  /// Extrae los factores clave que influyen en la predicción
  Map<String, String> _extractKeyFactors(
    Map<String, dynamic> features,
    double demand,
  ) {
    final factors = <String, String>{};

    factors['Estación'] = features['season'];
    factors['Familia'] = features['aggregated_family'];
    factors['Categoría'] = features['category'];
    factors['Precio'] = '\$${features['price'].toStringAsFixed(2)}';
    factors['Tiendas'] = '${features['num_stores'].toInt()}';
    factors['Demanda Estimada'] = '${demand.toInt()} unidades/semana';

    return factors;
  }
}

/// Resultado de una predicción con detalles adicionales
class PredictionResult {
  final double predictedDemand;
  final String season;
  final double confidence; // 0-100
  final Map<String, String> factors;

  PredictionResult({
    required this.predictedDemand,
    required this.season,
    required this.confidence,
    required this.factors,
  });

  /// Devuelve una interpretación textual de la demanda
  String get demandLevel {
    if (predictedDemand < 2000) return 'Baja';
    if (predictedDemand < 10000) return 'Media';
    if (predictedDemand < 50000) return 'Alta';
    return 'Muy Alta';
  }

  /// Devuelve un color asociado al nivel de demanda
  String get demandColor {
    if (predictedDemand < 2000) return 'red';
    if (predictedDemand < 10000) return 'orange';
    if (predictedDemand < 50000) return 'green';
    return 'blue';
  }
}

class _FeatureSpec {
  final String name;
  final bool isNumeric;

  const _FeatureSpec({required this.name, required this.isNumeric});
}

class _FeatureStat {
  final double mean;
  final double std;

  const _FeatureStat({required this.mean, required this.std});

  double get safeStd => std == 0 ? 1.0 : std;
}

class _Neighbor {
  final double distance;
  final double target;

  const _Neighbor({required this.distance, required this.target});
}
