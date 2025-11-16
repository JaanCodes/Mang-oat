import 'dart:convert';
import 'dart:io';
import 'package:http/http.dart' as http;
import 'package:mangoat/models/product_data.dart';

/// Resultado completo del análisis de IA
class AIAnalysisResult {
  final List<String>
  descriptiveTags; // Tags visuales/decorativos (10 tags detallados)
  final List<String> specificTags; // Tags técnicos para lógica interna
  final ProductData productData;
  final double weeklyDemand;
  final double confidence;

  AIAnalysisResult({
    required this.descriptiveTags,
    required this.specificTags,
    required this.productData,
    required this.weeklyDemand,
    required this.confidence,
  });
}

class GeminiService {
  // Lista de API keys con fallback
  static const List<String> _apiKeys = [
    '8461257c-261b-4eec-80c2-d6fcc03dac6f',
    '65cdb83d-db28-42be-8b7c-03d63132c62d',
    'cf98bb2c-3b13-4d45-b6e7-e353561be39d',
  ];
  static int _currentKeyIndex = 0;
  static const String _baseUrl = 'https://api.sambanova.ai/v1/chat/completions';
  static const String _model = 'Llama-4-Maverick-17B-128E-Instruct';

  static String get _apiKey => _apiKeys[_currentKeyIndex];

  static bool _tryNextApiKey() {
    if (_currentKeyIndex < _apiKeys.length - 1) {
      _currentKeyIndex++;
      return true;
    }
    return false;
  }

  static void _resetApiKeyIndex() {
    _currentKeyIndex = 0;
  }

  /// Análisis completo en una sola petición
  Future<AIAnalysisResult> analyzeComplete(File imageFile) async {
    _resetApiKeyIndex();

    while (true) {
      try {
        final bytes = await imageFile.readAsBytes();
        final base64Image = base64Encode(bytes);

        final now = DateTime.now();
        final currentMonth = now.month;

        final prompt =
            '''
Analiza esta imagen de prenda de ropa y responde con un JSON válido (sin markdown, sin explicaciones).

Estructura JSON requerida:
{
  "descriptive_tags": ["tag1", "tag2", "tag3", ..., "tag10"],
  "specific_tags": ["temporada", "categoria_tecnica", ...],
  "demand_factors": {
    "versatility": 1-10,
    "trend_level": 1-10,
    "seasonality_match": 1-10,
    "price_accessibility": 1-10,
    "target_audience_size": 1-10,
    "color_popularity": 1-10,
    "style_uniqueness": 1-10,
    "occasion_specificity": 1-10
  },
  "product": {
    "aggregated_family": "Woman|Man|Kids",
    "family": "Dresses|Shirts|Pants|Skirts|Jackets|Coats|Sweaters|T-Shirts|Jeans|Shoes|Bags|Accessories",
    "category": "Casual|Formal|Sport|Party|Beach|Work",
    "fabric": "Cotton|Polyester|Silk|Denim|Leather|Wool|Linen|Viscose",
    "color_name": "nombre_color",
    "length_type": "Short|Mid|Long|Maxi",
    "sleeve_length_type": "Sleeveless|Short|3/4|Long",
    "silhouette_type": "Straight|Fitted|Loose|A-line|Oversized",
    "waist_type": "High|Mid|Low|Natural",
    "neck_lapel_type": "Round|V-neck|Crew|Turtleneck|Collar|Hooded",
    "heel_shape_type": "Flat|Low|Mid|High|Platform",
    "toecap_type": "Round|Pointed|Square|Open",
    "woven_structure": "Plain|Twill|Satin|None",
    "knit_structure": "Jersey|Rib|Cable|None",
    "print_type": "Solid|Striped|Floral|Abstract|Geometric|Animal|Text",
    "archetype": "Classic|Trendy|Romantic|Sporty|Edgy|Minimalist",
    "moment": "Day|Night|Weekend|Work|Special",
    "estimated_price": 20-150,
    "estimated_stores": 50-300,
    "estimated_sizes": 3-8,
    "has_plus_sizes": 0|1,
    "estimated_lifecycle": 8-24
  }
}

DESCRIPTIVE_TAGS (10 tags visuales/decorativos para mostrar al usuario):
- Describe detalladamente la prenda con exactamente 10 tags creativos y específicos
- Incluye: colores exactos, patrones, texturas, detalles de diseño, estilo visual, ocasión de uso
- Ejemplos visuales: "azul marino", "botones dorados", "cuello redondo", "manga larga", "estampado floral", "textura suave", "estilo vintage", "casual elegante", "acabado brillante", "costuras contrastantes"
- Ejemplos para abrigo: "negro", "lana premium", "doble botonadura", "cuello alto", "bolsillos laterales", "forro interior", "corte recto", "elegante", "invierno", "formal"
- Sé muy descriptivo y específico en cada tag

SPECIFIC_TAGS (tags técnicos para lógica interna):
- Tags cortos para clasificación y estacionalidad
- DEBE incluir temporada: "invierno", "verano", "otoño", "primavera", o "todas_estaciones"
- Incluye: tipo básico, categoría funcional, público objetivo
- Ejemplos: ["invierno", "abrigo", "formal", "mujer"] o ["verano", "vestido", "casual", "mujer", "ligero"]

DEMAND_FACTORS (Factores de 1-10 para calcular demanda con ALTA VARIABILIDAD):
- **versatility**: ¿Cuán versátil es? (10=muy versátil como jeans negros, 1=muy específico como vestido de novia)
- **trend_level**: ¿Qué tan trendy/de moda es? (10=super trendy ahora, 1=clásico atemporal)
- **seasonality_match**: ¿Coincide con temporada actual (mes $currentMonth)? (10=perfecto para esta época, 1=totalmente fuera de temporada)
- **price_accessibility**: ¿Qué tan accesible es el precio? (10=muy barato/accesible, 1=muy caro/lujo)
- **target_audience_size**: ¿Qué tan grande es el público objetivo? (10=todo el mundo, 1=nicho muy específico)
- **color_popularity**: ¿Qué tan popular es el color? (10=negro/blanco/neutro, 1=color muy raro)
- **style_uniqueness**: ¿Qué tan único/específico es el estilo? (10=muy único/llamativo, 1=super básico)
- **occasion_specificity**: ¿Qué tan específica es la ocasión de uso? (10=muy específico como gala, 1=uso diario)

Estos factores se usarán para calcular la demanda con ALTA VARIABILIDAD. NO calcules weekly_current, solo proporciona estos factores del 1-10.

EJEMPLOS DE DEMAND_FACTORS:

1. Camiseta blanca básica algodón:
   versatility: 10, trend_level: 3, seasonality_match: 10, price_accessibility: 9,
   target_audience_size: 10, color_popularity: 10, style_uniqueness: 1, occasion_specificity: 1

2. Abrigo negro lana formal (Nov):
   versatility: 7, trend_level: 6, seasonality_match: 10, price_accessibility: 4,
   target_audience_size: 7, color_popularity: 9, style_uniqueness: 5, occasion_specificity: 6

3. Vestido floral verano ligero (Nov - fuera de temporada):
   versatility: 5, trend_level: 7, seasonality_match: 2, price_accessibility: 7,
   target_audience_size: 6, color_popularity: 6, style_uniqueness: 7, occasion_specificity: 5

4. Chaqueta dorada lentejuelas fiesta:
   versatility: 2, trend_level: 9, seasonality_match: 7, price_accessibility: 3,
   target_audience_size: 3, color_popularity: 2, style_uniqueness: 10, occasion_specificity: 10

5. Jeans azul clásico:
   versatility: 10, trend_level: 5, seasonality_match: 10, price_accessibility: 8,
   target_audience_size: 10, color_popularity: 9, style_uniqueness: 2, occasion_specificity: 1

Sé MUY ESPECÍFICO con los valores 1-10. NO uses siempre 5 o valores medios. Analiza cada característica independientemente.

Responde ÚNICAMENTE con el JSON, sin ```json ni explicaciones.
''';

        final requestBody = {
          'stream': false,
          'model': _model,
          'messages': [
            {
              'role': 'user',
              'content': [
                {'type': 'text', 'text': prompt},
                {
                  'type': 'image_url',
                  'image_url': {'url': 'data:image/jpeg;base64,$base64Image'},
                },
              ],
            },
          ],
        };

        final response = await http.post(
          Uri.parse(_baseUrl),
          headers: {
            'Authorization': 'Bearer $_apiKey',
            'Content-Type': 'application/json',
          },
          body: jsonEncode(requestBody),
        );

        if (response.statusCode == 200) {
          final jsonResponse = jsonDecode(response.body);
          final text =
              jsonResponse['choices'][0]['message']['content'] as String;

          return _parseAIResponse(text);
        } else {
          throw Exception('API error: ${response.statusCode}');
        }
      } catch (e) {
        if (_tryNextApiKey()) {
          continue;
        } else {
          return _generateFallback();
        }
      }
    }
  }

  AIAnalysisResult _parseAIResponse(String text) {
    try {
      String cleanJson = text.trim();
      if (cleanJson.startsWith('```json')) {
        cleanJson = cleanJson.substring(7);
      }
      if (cleanJson.startsWith('```')) {
        cleanJson = cleanJson.substring(3);
      }
      if (cleanJson.endsWith('```')) {
        cleanJson = cleanJson.substring(0, cleanJson.length - 3);
      }
      cleanJson = cleanJson.trim();

      final data = jsonDecode(cleanJson);

      final descriptiveTags =
          (data['descriptive_tags'] as List<dynamic>?)
              ?.map((e) => e.toString())
              .toList() ??
          [
            'prenda',
            'moda',
            'básico',
            'versátil',
            'cómodo',
            'casual',
            'diario',
            'estándar',
            'simple',
            'funcional',
          ];

      final specificTags =
          (data['specific_tags'] as List<dynamic>?)
              ?.map((e) => e.toString())
              .toList() ??
          ['todas_estaciones', 'casual', 'básico'];

      final product = data['product'] ?? {};
      final demandFactors = data['demand_factors'] ?? {};

      final now = DateTime.now();
      final seasonCode = _getSeasonCode(now);

      final productData = ProductData(
        lifeCycleLength:
            (product['estimated_lifecycle'] as num?)?.toDouble() ?? 12.0,
        numStores: (product['estimated_stores'] as num?)?.toDouble() ?? 150.0,
        numSizes: (product['estimated_sizes'] as num?)?.toDouble() ?? 5.0,
        hasPlusSizes: (product['has_plus_sizes'] as num?)?.toDouble() ?? 1.0,
        price: (product['estimated_price'] as num?)?.toDouble() ?? 29.99,
        idSeason: seasonCode,
        aggregatedFamily: product['aggregated_family'] as String? ?? 'Woman',
        family: product['family'] as String? ?? 'Unknown',
        category: product['category'] as String? ?? 'Casual',
        fabric: product['fabric'] as String? ?? 'Cotton',
        colorName: product['color_name'] as String? ?? 'Unknown',
        lengthType: product['length_type'] as String? ?? 'Mid',
        silhouetteType: product['silhouette_type'] as String? ?? 'Straight',
        waistType: product['waist_type'] as String? ?? 'Natural',
        neckLapelType: product['neck_lapel_type'] as String? ?? 'Round',
        sleeveLengthType: product['sleeve_length_type'] as String? ?? 'Short',
        heelShapeType: product['heel_shape_type'] as String? ?? 'Flat',
        toecapType: product['toecap_type'] as String? ?? 'Round',
        wovenStructure: product['woven_structure'] as String? ?? 'Plain',
        knitStructure: product['knit_structure'] as String? ?? 'Jersey',
        printType: product['print_type'] as String? ?? 'Solid',
        archetype: product['archetype'] as String? ?? 'Classic',
        moment: product['moment'] as String? ?? 'Day',
        phaseIn: now,
        imageEmbedding: _generateEmbedding(product),
      );

      // Calcular demanda basada en los factores con alta variabilidad
      final weeklyDemand = _calculateDemandFromFactors(
        demandFactors,
        product,
        specificTags,
      );

      final confidence = _calculateConfidenceFromFactors(demandFactors);

      return AIAnalysisResult(
        descriptiveTags: descriptiveTags,
        specificTags: specificTags,
        productData: productData,
        weeklyDemand: weeklyDemand,
        confidence: confidence,
      );
    } catch (e) {
      return _generateFallback();
    }
  }

  AIAnalysisResult _generateFallback() {
    final now = DateTime.now();

    final descriptiveTags = [
      'prenda',
      'casual',
      'básico',
      'cómodo',
      'versátil',
      'diario',
      'simple',
      'funcional',
      'estándar',
      'clásico',
    ];
    final specificTags = ['todas_estaciones', 'casual', 'básico'];

    final productData = ProductData(
      lifeCycleLength: 12.0,
      numStores: 150.0,
      numSizes: 5.0,
      hasPlusSizes: 1.0,
      price: 29.99,
      idSeason: _getSeasonCode(now),
      aggregatedFamily: 'Woman',
      family: 'T-Shirts',
      category: 'Casual',
      fabric: 'Cotton',
      colorName: 'Neutral',
      lengthType: 'Mid',
      silhouetteType: 'Straight',
      waistType: 'Natural',
      neckLapelType: 'Round',
      sleeveLengthType: 'Short',
      heelShapeType: 'Flat',
      toecapType: 'Round',
      wovenStructure: 'Plain',
      knitStructure: 'Jersey',
      printType: 'Solid',
      archetype: 'Classic',
      moment: 'Day',
      phaseIn: now,
      imageEmbedding: List.filled(256, 0.0),
    );

    final weeklyDemand = 5000.0;
    final confidence = 70.0;

    return AIAnalysisResult(
      descriptiveTags: descriptiveTags,
      specificTags: specificTags,
      productData: productData,
      weeklyDemand: weeklyDemand,
      confidence: confidence,
    );
  }

  /// Calcula la demanda semanal basada en los factores de demanda
  double _calculateDemandFromFactors(
    Map<String, dynamic> factors,
    Map<String, dynamic> product,
    List<String> specificTags,
  ) {
    // Extraer factores (1-10)
    final versatility = (factors['versatility'] as num?)?.toDouble() ?? 5.0;
    final trendLevel = (factors['trend_level'] as num?)?.toDouble() ?? 5.0;
    final seasonalityMatch =
        (factors['seasonality_match'] as num?)?.toDouble() ?? 5.0;
    final priceAccessibility =
        (factors['price_accessibility'] as num?)?.toDouble() ?? 5.0;
    final targetAudienceSize =
        (factors['target_audience_size'] as num?)?.toDouble() ?? 5.0;
    final colorPopularity =
        (factors['color_popularity'] as num?)?.toDouble() ?? 5.0;
    final styleUniqueness =
        (factors['style_uniqueness'] as num?)?.toDouble() ?? 5.0;
    final occasionSpecificity =
        (factors['occasion_specificity'] as num?)?.toDouble() ?? 5.0;

    // Demanda base (varía según familia)
    double baseDemand = 5000.0;
    final family = product['aggregated_family'] as String? ?? 'Woman';
    if (family == 'Woman')
      baseDemand = 6000.0;
    else if (family == 'Man')
      baseDemand = 5000.0;
    else if (family == 'Kids')
      baseDemand = 4000.0;

    // Multiplicadores basados en factores (con alta variabilidad)
    // Versatilidad y público objetivo son los más importantes (peso alto)
    final versatilityMultiplier = 0.5 + (versatility / 10.0) * 1.5; // 0.5 a 2.0
    final audienceMultiplier =
        0.4 + (targetAudienceSize / 10.0) * 1.6; // 0.4 a 2.0

    // Estacionalidad tiene impacto muy fuerte
    final seasonalityMultiplier =
        0.3 + (seasonalityMatch / 10.0) * 1.4; // 0.3 a 1.7

    // Precio accesible aumenta demanda significativamente
    final priceMultiplier =
        0.6 + (priceAccessibility / 10.0) * 1.2; // 0.6 a 1.8

    // Color popular aumenta demanda
    final colorMultiplier = 0.7 + (colorPopularity / 10.0) * 0.8; // 0.7 a 1.5

    // Prendas muy únicas tienen menos demanda (relación inversa)
    final uniquenessMultiplier =
        1.5 - (styleUniqueness / 10.0) * 0.8; // 0.7 a 1.5

    // Ocasiones específicas reducen demanda (relación inversa)
    final occasionMultiplier =
        1.4 - (occasionSpecificity / 10.0) * 0.7; // 0.7 a 1.4

    // Tendencias moderadamente importantes
    final trendMultiplier = 0.8 + (trendLevel / 10.0) * 0.6; // 0.8 a 1.4

    // Ajustes por características del producto
    final numStores =
        (product['estimated_stores'] as num?)?.toDouble() ?? 150.0;
    final storesMultiplier = 0.7 + (numStores / 300.0) * 0.8; // 0.7 a 1.5

    final numSizes = (product['estimated_sizes'] as num?)?.toDouble() ?? 5.0;
    final sizesMultiplier = 0.85 + (numSizes / 8.0) * 0.4; // 0.85 a 1.25

    final hasPlusSizes = (product['has_plus_sizes'] as num?)?.toDouble() ?? 0.0;
    final plusSizesMultiplier = 1.0 + (hasPlusSizes * 0.15); // 1.0 a 1.15

    // Calcular demanda con todos los multiplicadores
    double demand =
        baseDemand *
        versatilityMultiplier *
        audienceMultiplier *
        seasonalityMultiplier *
        priceMultiplier *
        colorMultiplier *
        uniquenessMultiplier *
        occasionMultiplier *
        trendMultiplier *
        storesMultiplier *
        sizesMultiplier *
        plusSizesMultiplier;

    // Añadir variabilidad aleatoria basada en factores únicos
    final uniqueHash =
        versatility.hashCode ^
        trendLevel.hashCode ^
        seasonalityMatch.hashCode ^
        styleUniqueness.hashCode;
    final randomSeed = (uniqueHash % 1000) / 1000.0;
    final randomMultiplier = 0.75 + (randomSeed * 0.5); // 0.75 a 1.25
    demand *= randomMultiplier;

    // Limitar al rango realista
    return demand.clamp(500.0, 9500.0);
  }

  /// Calcula la confianza basada en los factores
  double _calculateConfidenceFromFactors(Map<String, dynamic> factors) {
    final versatility = (factors['versatility'] as num?)?.toDouble() ?? 5.0;
    final seasonalityMatch =
        (factors['seasonality_match'] as num?)?.toDouble() ?? 5.0;
    final targetAudienceSize =
        (factors['target_audience_size'] as num?)?.toDouble() ?? 5.0;
    final styleUniqueness =
        (factors['style_uniqueness'] as num?)?.toDouble() ?? 5.0;

    // Base de confianza
    double confidence = 70.0;

    // Mayor confianza para prendas versátiles y con buen match estacional
    confidence += (versatility - 5.0) * 2.0; // ±10
    confidence += (seasonalityMatch - 5.0) * 2.5; // ±12.5
    confidence += (targetAudienceSize - 5.0) * 1.5; // ±7.5

    // Menor confianza para prendas muy únicas (más impredecibles)
    confidence -= (styleUniqueness - 5.0) * 1.0; // ±5

    return confidence.clamp(60.0, 95.0);
  }

  int _getSeasonCode(DateTime date) {
    final month = date.month;
    if (month >= 12 || month <= 2) {
      return date.year * 100 + 1;
    } else if (month >= 3 && month <= 5) {
      return date.year * 100 + 2;
    } else if (month >= 6 && month <= 8) {
      return date.year * 100 + 3;
    } else {
      return date.year * 100 + 4;
    }
  }

  List<double> _generateEmbedding(Map<String, dynamic> productJson) {
    final random = productJson.hashCode;
    final embedding = List<double>.filled(256, 0.0);
    for (int i = 0; i < 256; i++) {
      embedding[i] = ((random + i * 17) % 100) / 100.0 - 0.5;
    }
    return embedding;
  }
}
