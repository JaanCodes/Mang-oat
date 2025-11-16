import 'dart:convert';
import 'dart:io';
import 'package:http/http.dart' as http;
import 'package:mangoat/models/product_data.dart';

class GeminiService {
  // Lista de API keys con fallback
  static const List<String> _apiKeys = [
    '65cdb83d-db28-42be-8b7c-03d63132c62d',
    'cf98bb2c-3b13-4d45-b6e7-e353561be39d',
  ];
  static int _currentKeyIndex = 0;
  static const String _baseUrl = 'https://api.sambanova.ai/v1/chat/completions';
  static const String _model = 'Llama-4-Maverick-17B-128E-Instruct';

  /// Obtiene la API key actual
  static String get _apiKey => _apiKeys[_currentKeyIndex];

  /// Intenta cambiar a la siguiente API key disponible
  static bool _tryNextApiKey() {
    if (_currentKeyIndex < _apiKeys.length - 1) {
      _currentKeyIndex++;
      print(
        'Cambiando a API key alternativa (${_currentKeyIndex + 1}/${_apiKeys.length})',
      );
      return true;
    }
    return false;
  }

  /// Reinicia el índice de API keys al inicio
  static void _resetApiKeyIndex() {
    _currentKeyIndex = 0;
  }

  Future<List<String>> analyzeClothing(File imageFile) async {
    _resetApiKeyIndex(); // Reiniciar al inicio de cada operación

    while (true) {
      try {
        // Leer la imagen y convertirla a base64
        final bytes = await imageFile.readAsBytes();
        final base64Image = base64Encode(bytes);

        // Crear el prompt para análisis de ropa
        final prompt = '''
Analiza esta imagen de una prenda de ropa y proporciona una lista de tags descriptivos.
Los tags deben incluir:
- Tipo de prenda (camisa, pantalón, vestido, etc.)
- Colores principales
- Estilo (casual, formal, deportivo, etc.)
- Material o textura aparente
- Características especiales (estampados, diseños, etc.)
- Temporada sugerida

Devuelve SOLO una lista de palabras clave separadas por comas, sin explicaciones adicionales.
Ejemplo: camiseta, azul, algodón, casual, manga corta, verano, lisa
''';

        // Preparar el body de la solicitud
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

        // Realizar la solicitud
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

          // Extraer el texto de la respuesta
          final text =
              jsonResponse['choices'][0]['message']['content'] as String;

          // Procesar el texto para obtener los tags
          final tags = text
              .trim()
              .split(',')
              .map((tag) => tag.trim())
              .where((tag) => tag.isNotEmpty)
              .toList();

          return tags;
        } else {
          throw Exception(
            'Error en la API: ${response.statusCode} - ${response.body}',
          );
        }
      } catch (e) {
        // Si hay error y hay más keys disponibles, intentar con la siguiente
        if (_tryNextApiKey()) {
          continue; // Reintentar con la siguiente key
        } else {
          // No hay más keys, lanzar el error
          throw Exception('Error al analizar la imagen: $e');
        }
      }
    }
  }

  Future<String> getDetailedDescription(File imageFile) async {
    _resetApiKeyIndex(); // Reiniciar al inicio de cada operación

    while (true) {
      try {
        final bytes = await imageFile.readAsBytes();
        final base64Image = base64Encode(bytes);

        final prompt = '''
Describe esta prenda de ropa de manera detallada y atractiva, como si fueras un experto en moda.
Incluye información sobre el estilo, cómo combinarla y para qué ocasiones es ideal.
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
          return jsonResponse['choices'][0]['message']['content'] as String;
        } else {
          throw Exception('Error en la API: ${response.statusCode}');
        }
      } catch (e) {
        // Si hay error y hay más keys disponibles, intentar con la siguiente
        if (_tryNextApiKey()) {
          continue; // Reintentar con la siguiente key
        } else {
          // No hay más keys, lanzar el error
          throw Exception('Error al obtener descripción: $e');
        }
      }
    }
  }

  /// Extrae información estructurada del producto para el modelo de predicción
  Future<ProductData> extractProductData(File imageFile) async {
    _resetApiKeyIndex(); // Reiniciar al inicio de cada operación

    while (true) {
      try {
        final bytes = await imageFile.readAsBytes();
        final base64Image = base64Encode(bytes);

        final prompt = '''
Analiza esta imagen de ropa detenidamente y extrae la siguiente información en formato JSON.
Sé lo más específico y preciso posible:

{
  "aggregated_family": "Woman" o "Man" o "Kids" (determina para quién es la prenda),
  "family": "Dresses" o "Shirts" o "Pants" o "Skirts" o "Jackets" o "Coats" o "Sweaters" o "T-Shirts" o "Jeans" o "Shoes" o "Bags" o "Accessories",
  "category": "Casual" o "Formal" o "Sport" o "Party" o "Beach" o "Work",
  "fabric": "Cotton" o "Polyester" o "Silk" o "Denim" o "Leather" o "Wool" o "Linen" o "Viscose" (estima el material),
  "color_name": color principal visible (ej: "Blue", "Black", "Red", "White", etc),
  "length_type": "Short" o "Mid" o "Long" o "Maxi" (para prendas con largo definido),
  "sleeve_length_type": "Sleeveless" o "Short" o "3/4" o "Long" (para prendas con mangas),
  "silhouette_type": "Straight" o "Fitted" o "Loose" o "A-line" o "Oversized",
  "waist_type": "High" o "Mid" o "Low" o "Natural" (para pantalones/faldas),
  "neck_lapel_type": "Round" o "V-neck" o "Crew" o "Turtleneck" o "Collar" o "Hooded",
  "heel_shape_type": "Flat" o "Low" o "Mid" o "High" o "Platform" (solo para zapatos),
  "toecap_type": "Round" o "Pointed" o "Square" o "Open" (solo para zapatos),
  "woven_structure": "Plain" o "Twill" o "Satin" o "None",
  "knit_structure": "Jersey" o "Rib" o "Cable" o "None",
  "print_type": "Solid" o "Striped" o "Floral" o "Abstract" o "Geometric" o "Animal" o "Text",
  "archetype": "Classic" o "Trendy" o "Romantic" o "Sporty" o "Edgy" o "Minimalist",
  "moment": "Day" o "Night" o "Weekend" o "Work" o "Special",
  "estimated_price": precio estimado en dólares USD (20-150 típicamente),
  "estimated_stores": número estimado de tiendas donde se vendería (50-300),
  "estimated_sizes": número de tallas disponibles (3-8),
  "has_plus_sizes": 1 si probablemente tiene tallas grandes, 0 si no,
  "estimated_lifecycle": duración estimada en el catálogo en semanas (8-24)
}

IMPORTANTE: Responde ÚNICAMENTE con el objeto JSON válido, sin markdown, sin explicaciones.
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

          // Intentar parsear el JSON de la respuesta
          try {
            // Limpiar el texto para obtener solo el JSON
            String cleanedJson = text.trim();
            if (cleanedJson.startsWith('```json')) {
              cleanedJson = cleanedJson.substring(7);
            }
            if (cleanedJson.startsWith('```')) {
              cleanedJson = cleanedJson.substring(3);
            }
            if (cleanedJson.endsWith('```')) {
              cleanedJson = cleanedJson.substring(0, cleanedJson.length - 3);
            }
            cleanedJson = cleanedJson.trim();

            final productJson = jsonDecode(cleanedJson);

            // Determinar la temporada actual para el season code
            final now = DateTime.now();
            final month = now.month;
            int seasonCode;
            if (month >= 12 || month <= 2) {
              seasonCode = now.year * 100 + 1; // Winter
            } else if (month >= 3 && month <= 5) {
              seasonCode = now.year * 100 + 2; // Spring
            } else if (month >= 6 && month <= 8) {
              seasonCode = now.year * 100 + 3; // Summer
            } else {
              seasonCode = now.year * 100 + 4; // Autumn
            }

            // Crear ProductData con los valores extraídos por la IA
            return ProductData(
              lifeCycleLength:
                  (productJson['estimated_lifecycle'] as num?)?.toDouble() ??
                  12.0,
              numStores:
                  (productJson['estimated_stores'] as num?)?.toDouble() ??
                  150.0,
              numSizes:
                  (productJson['estimated_sizes'] as num?)?.toDouble() ?? 5.0,
              hasPlusSizes:
                  (productJson['has_plus_sizes'] as num?)?.toDouble() ?? 1.0,
              price:
                  (productJson['estimated_price'] as num?)?.toDouble() ?? 29.99,
              idSeason: seasonCode,
              aggregatedFamily:
                  productJson['aggregated_family'] as String? ?? 'Woman',
              family: productJson['family'] as String? ?? 'Unknown',
              category: productJson['category'] as String? ?? 'Casual',
              fabric: productJson['fabric'] as String? ?? 'Cotton',
              colorName: productJson['color_name'] as String? ?? 'Unknown',
              lengthType: productJson['length_type'] as String? ?? 'Mid',
              silhouetteType:
                  productJson['silhouette_type'] as String? ?? 'Straight',
              waistType: productJson['waist_type'] as String? ?? 'Natural',
              neckLapelType:
                  productJson['neck_lapel_type'] as String? ?? 'Round',
              sleeveLengthType:
                  productJson['sleeve_length_type'] as String? ?? 'Short',
              heelShapeType:
                  productJson['heel_shape_type'] as String? ?? 'Flat',
              toecapType: productJson['toecap_type'] as String? ?? 'Round',
              wovenStructure:
                  productJson['woven_structure'] as String? ?? 'Plain',
              knitStructure:
                  productJson['knit_structure'] as String? ?? 'Jersey',
              printType: productJson['print_type'] as String? ?? 'Solid',
              archetype: productJson['archetype'] as String? ?? 'Classic',
              moment: productJson['moment'] as String? ?? 'Day',
              phaseIn: DateTime.now(),
              imageEmbedding: _generateSyntheticEmbedding(productJson),
            );
          } catch (e) {
            // Si falla el parseo, usar valores por defecto
            print('Error al parsear JSON del producto: $e');
            return ProductData.example();
          }
        } else {
          throw Exception('Error en la API: ${response.statusCode}');
        }
      } catch (e) {
        // Si hay error y hay más keys disponibles, intentar con la siguiente
        if (_tryNextApiKey()) {
          continue; // Reintentar con la siguiente key
        } else {
          // No hay más keys, retornar datos de ejemplo
          print('Error al extraer datos del producto: $e');
          return ProductData.example();
        }
      }
    }
  }

  /// Genera predicciones de demanda mensuales usando IA considerando estacionalidad
  Future<List<double>> predictMonthlyDemandWithAI(
    ProductData productData,
  ) async {
    _resetApiKeyIndex();

    // Determinar el mes actual
    final now = DateTime.now();
    final currentMonth = now.month;

    // Crear un contexto detallado del producto
    final productContext =
        '''
Producto:
- Tipo: ${productData.family}
- Categoría: ${productData.category}
- Familia: ${productData.aggregatedFamily}
- Material: ${productData.fabric}
- Estilo: ${productData.archetype}
- Momento: ${productData.moment}
- Tipo de manga: ${productData.sleeveLengthType}
- Longitud: ${productData.lengthType}
- Precio: \$${productData.price.toStringAsFixed(2)}
- Ciclo de vida: ${productData.lifeCycleLength.toInt()} semanas
- Número de tiendas: ${productData.numStores.toInt()}
''';

    final prompt =
        '''
Como experto en análisis de demanda de moda, genera predicciones realistas de demanda semanal para los próximos 12 meses considerando la estacionalidad.

$productContext

Mes actual: Mes $currentMonth (${_getMonthName(currentMonth)})

IMPORTANTE: Considera estos factores estacionales:
- Ropa de invierno (abrigos, lana, manga larga): mayor demanda Nov-Feb, baja demanda Jun-Ago
- Ropa de verano (shorts, sin mangas, ligera): mayor demanda May-Ago, baja demanda Dic-Feb
- Ropa de primavera/otoño (entretiempo): picos en Mar-Abr y Sep-Oct
- Ropa todo terreno (jeans, básicos): demanda estable todo el año con ligeras variaciones

Genera 12 números (uno por cada mes empezando desde el mes actual) que representen la demanda semanal esperada en unidades.
Los valores deben:
1. Ser realistas (típicamente entre 1000-15000 unidades/semana)
2. Reflejar patrones estacionales naturales según el tipo de prenda
3. Considerar el precio (más caro = menor demanda)
4. Considerar el número de tiendas (más tiendas = mayor demanda)

Responde ÚNICAMENTE con 12 números separados por comas, sin explicaciones.
Ejemplo: 8500,9200,12000,11500,8000,5500,4200,4500,7800,10200,11800,9500
''';

    while (true) {
      try {
        final requestBody = {
          'stream': false,
          'model': _model,
          'messages': [
            {'role': 'user', 'content': prompt},
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

          // Parsear los números
          final cleanText = text.trim().replaceAll(RegExp(r'[^\d,.]'), '');
          final values = cleanText
              .split(',')
              .map((s) => double.tryParse(s.trim()) ?? 5000.0)
              .toList();

          // Asegurar que tenemos exactamente 12 valores
          while (values.length < 12) {
            values.add(5000.0);
          }
          if (values.length > 12) {
            return values.sublist(0, 12);
          }

          return values;
        } else {
          throw Exception('Error en la API: ${response.statusCode}');
        }
      } catch (e) {
        if (_tryNextApiKey()) {
          continue;
        } else {
          // Fallback: generar valores por defecto con patrón estacional básico
          print('Error en predicción IA: $e. Usando fallback.');
          return _generateSeasonalFallback(productData, currentMonth);
        }
      }
    }
  }

  /// Genera un fallback con patrón estacional básico
  List<double> _generateSeasonalFallback(
    ProductData productData,
    int currentMonth,
  ) {
    final baseDemand = 5000.0;
    final predictions = <double>[];

    // Determinar si es ropa de invierno, verano o todo terreno
    final isWinterClothing =
        productData.fabric.toLowerCase().contains('wool') ||
        productData.fabric.toLowerCase().contains('leather') ||
        productData.sleeveLengthType.toLowerCase().contains('long') ||
        productData.family.toLowerCase().contains('coat') ||
        productData.family.toLowerCase().contains('jacket');

    final isSummerClothing =
        productData.sleeveLengthType.toLowerCase().contains('sleeveless') ||
        productData.sleeveLengthType.toLowerCase().contains('short') ||
        productData.lengthType.toLowerCase().contains('short') ||
        productData.family.toLowerCase().contains('short') ||
        productData.category.toLowerCase().contains('beach');

    for (int i = 0; i < 12; i++) {
      final month = ((currentMonth + i - 1) % 12) + 1;
      double seasonalMultiplier = 1.0;

      if (isWinterClothing) {
        // Más demanda en invierno (Nov-Feb)
        if (month >= 11 || month <= 2) {
          seasonalMultiplier = 1.5;
        } else if (month >= 6 && month <= 8) {
          seasonalMultiplier = 0.4;
        }
      } else if (isSummerClothing) {
        // Más demanda en verano (May-Ago)
        if (month >= 5 && month <= 8) {
          seasonalMultiplier = 1.6;
        } else if (month >= 12 || month <= 2) {
          seasonalMultiplier = 0.3;
        }
      } else {
        // Ropa todo terreno con ligeras variaciones
        if (month >= 3 && month <= 5 || month >= 9 && month <= 10) {
          seasonalMultiplier = 1.2; // Primavera/Otoño
        }
      }

      predictions.add(baseDemand * seasonalMultiplier);
    }

    return predictions;
  }

  String _getMonthName(int month) {
    const months = [
      'Enero',
      'Febrero',
      'Marzo',
      'Abril',
      'Mayo',
      'Junio',
      'Julio',
      'Agosto',
      'Septiembre',
      'Octubre',
      'Noviembre',
      'Diciembre',
    ];
    return months[month - 1];
  }

  /// Genera un embedding sintético basado en las características del producto
  /// Esto simula el vector de imagen que normalmente vendría de una red neuronal
  List<double> _generateSyntheticEmbedding(Map<String, dynamic> productJson) {
    final random =
        productJson.hashCode; // Usar hash como seed para consistencia
    final embedding = List<double>.filled(256, 0.0);

    // Generar valores pseudo-aleatorios basados en las características
    for (int i = 0; i < 256; i++) {
      embedding[i] =
          ((random + i * 17) % 100) / 100.0 - 0.5; // Valores entre -0.5 y 0.5
    }

    return embedding;
  }
}
