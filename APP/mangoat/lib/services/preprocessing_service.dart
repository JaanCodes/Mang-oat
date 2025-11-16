import 'dart:math';
import 'package:mangoat/models/product_data.dart';

/// Servicio para preprocesar datos de productos antes de la predicción
class PreprocessingService {
  // Mapeo de estaciones (equivalente a Python)
  static const Map<int, String> seasonMap = {
    1: 'Winter',
    2: 'Winter',
    3: 'Spring',
    4: 'Spring',
    5: 'Spring',
    6: 'Summer',
    7: 'Summer',
    8: 'Summer',
    9: 'Autumn',
    10: 'Autumn',
    11: 'Autumn',
    12: 'Winter',
  };

  /// Calcula las features de Fourier (sin/cos) para el mes
  static Map<String, double> calculateFourierFeatures(DateTime? date) {
    int month;
    if (date == null) {
      month = 7; // Valor por defecto (verano)
    } else {
      month = date.month;
    }

    final monthNormalized = month.toDouble();
    final monthSin = sin(2 * pi * monthNormalized / 12);
    final monthCos = cos(2 * pi * monthNormalized / 12);

    return {'month_sin': monthSin, 'month_cos': monthCos};
  }

  /// Determina la estación del año basándose en el mes
  static String getSeason(DateTime? date) {
    int month;
    if (date == null) {
      month = 7; // Valor por defecto (verano)
    } else {
      month = date.month;
    }

    return seasonMap[month] ?? 'Unknown';
  }

  /// Crea features de interacción entre atributos y temporadas
  static Map<String, String> createTrendFeatures(
    ProductData product,
    String season,
  ) {
    return {
      'sleeve_length_type_X_season': '${product.sleeveLengthType}_S_$season',
      'family_X_season': '${product.family}_S_$season',
      'fabric_X_season': '${product.fabric}_S_$season',
      'length_type_X_season': '${product.lengthType}_S_$season',
    };
  }

  /// Calcula similitud coseno entre dos vectores
  static double cosineSimilarity(List<double> a, List<double> b) {
    if (a.length != b.length) {
      throw ArgumentError('Los vectores deben tener la misma longitud');
    }

    double dotProduct = 0.0;
    double normA = 0.0;
    double normB = 0.0;

    for (int i = 0; i < a.length; i++) {
      dotProduct += a[i] * b[i];
      normA += a[i] * a[i];
      normB += b[i] * b[i];
    }

    if (normA == 0.0 || normB == 0.0) {
      return 0.0;
    }

    return dotProduct / (sqrt(normA) * sqrt(normB));
  }

  /// Calcula distancia euclidiana entre dos vectores
  static double euclideanDistance(List<double> a, List<double> b) {
    if (a.length != b.length) {
      throw ArgumentError('Los vectores deben tener la misma longitud');
    }

    double sum = 0.0;
    for (int i = 0; i < a.length; i++) {
      final diff = a[i] - b[i];
      sum += diff * diff;
    }

    return sqrt(sum);
  }

  /// Crea features de embeddings estacionales (simplificado)
  /// En producción, estos valores vendrían de estadísticas pre-calculadas
  static Map<String, double> createEmbeddingFeatures(
    List<double> embedding,
    int idSeason,
  ) {
    // Para una implementación completa, necesitaríamos los centroides
    // y clusters precalculados. Por ahora, usamos valores por defecto
    // que serán actualizados con datos reales.

    return {
      'embedding_cluster': -1.0,
      'similarity_to_season_center': 0.5, // Valor neutro
      'avg_distance_within_season': 1.0,
      'similarity_to_nearest_seasons': 0.5,
    };
  }

  /// Normaliza un valor numérico usando media y desviación estándar
  static double normalizeValue(double value, double mean, double scale) {
    if (scale == 0.0) return 0.0;
    return (value - mean) / scale;
  }

  /// Procesa un producto completo y devuelve todas las features necesarias
  static Map<String, dynamic> processProduct(ProductData product) {
    // 1. Features de Fourier (temporales)
    final fourierFeatures = calculateFourierFeatures(product.phaseIn);

    // 2. Determinar estación
    final season = getSeason(product.phaseIn);

    // 3. Features de tendencia (interacciones)
    final trendFeatures = createTrendFeatures(product, season);

    // 4. Features de embeddings estacionales
    final embeddingFeatures = createEmbeddingFeatures(
      product.imageEmbedding,
      product.idSeason,
    );

    // 5. Compilar todas las features
    return {
      // Numéricas básicas
      'life_cycle_length': product.lifeCycleLength,
      'num_stores': product.numStores,
      'num_sizes': product.numSizes,
      'has_plus_sizes': product.hasPlusSizes,
      'price': product.price,

      // Features de Fourier
      'month_sin': fourierFeatures['month_sin']!,
      'month_cos': fourierFeatures['month_cos']!,

      // Features de embeddings
      'similarity_to_season_center':
          embeddingFeatures['similarity_to_season_center']!,
      'avg_distance_within_season':
          embeddingFeatures['avg_distance_within_season']!,
      'similarity_to_nearest_seasons':
          embeddingFeatures['similarity_to_nearest_seasons']!,
      'embedding_cluster': embeddingFeatures['embedding_cluster']!,

      // Categóricas
      'id_season': product.idSeason,
      'aggregated_family': product.aggregatedFamily,
      'family': product.family,
      'category': product.category,
      'fabric': product.fabric,
      'color_name': product.colorName,
      'length_type': product.lengthType,
      'silhouette_type': product.silhouetteType,
      'waist_type': product.waistType,
      'neck_lapel_type': product.neckLapelType,
      'sleeve_length_type': product.sleeveLengthType,
      'heel_shape_type': product.heelShapeType,
      'toecap_type': product.toecapType,
      'woven_structure': product.wovenStructure,
      'knit_structure': product.knitStructure,
      'print_type': product.printType,
      'archetype': product.archetype,
      'moment': product.moment,
      'season': season,

      // Features de tendencia
      'sleeve_length_type_X_season':
          trendFeatures['sleeve_length_type_X_season']!,
      'family_X_season': trendFeatures['family_X_season']!,
      'fabric_X_season': trendFeatures['fabric_X_season']!,
      'length_type_X_season': trendFeatures['length_type_X_season']!,

      // Embedding original (para PCA)
      'image_embedding': product.imageEmbedding,
    };
  }

  /// Aplica PCA a un embedding usando componentes precalculados
  static List<double> applyPCA(
    List<double> embedding,
    List<double> pcaMean,
    List<List<double>> pcaComponents,
  ) {
    // Centrar el embedding restando la media
    final centered = List.generate(
      embedding.length,
      (i) => embedding[i] - pcaMean[i],
    );

    // Proyectar en los componentes principales
    final projected = <double>[];
    for (final component in pcaComponents) {
      double dotProduct = 0.0;
      for (int i = 0; i < centered.length; i++) {
        dotProduct += centered[i] * component[i];
      }
      projected.add(dotProduct);
    }

    return projected;
  }

  /// Normaliza las features numéricas usando estadísticas del entrenamiento
  static Map<String, double> normalizeNumericFeatures(
    Map<String, double> features,
    Map<String, double> means,
    Map<String, double> scales,
  ) {
    final normalized = <String, double>{};

    features.forEach((key, value) {
      final mean = means[key] ?? 0.0;
      final scale = scales[key] ?? 1.0;
      normalized[key] = normalizeValue(value, mean, scale);
    });

    return normalized;
  }
}
