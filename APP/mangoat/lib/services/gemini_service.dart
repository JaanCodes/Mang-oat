import 'dart:convert';
import 'dart:io';
import 'package:http/http.dart' as http;

class GeminiService {
  static const String _apiKey = '65cdb83d-db28-42be-8b7c-03d63132c62d';
  static const String _baseUrl = 'https://api.sambanova.ai/v1/chat/completions';
  static const String _model = 'Llama-4-Maverick-17B-128E-Instruct';

  Future<List<String>> analyzeClothing(File imageFile) async {
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
              {
                'type': 'text',
                'text': prompt,
              },
              {
                'type': 'image_url',
                'image_url': {
                  'url': 'data:image/jpeg;base64,$base64Image',
                }
              }
            ]
          }
        ]
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
        final text = jsonResponse['choices'][0]['message']['content'] as String;
        
        // Procesar el texto para obtener los tags
        final tags = text
            .trim()
            .split(',')
            .map((tag) => tag.trim())
            .where((tag) => tag.isNotEmpty)
            .toList();

        return tags;
      } else {
        throw Exception('Error en la API: ${response.statusCode} - ${response.body}');
      }
    } catch (e) {
      throw Exception('Error al analizar la imagen: $e');
    }
  }

  Future<String> getDetailedDescription(File imageFile) async {
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
              {
                'type': 'text',
                'text': prompt,
              },
              {
                'type': 'image_url',
                'image_url': {
                  'url': 'data:image/jpeg;base64,$base64Image',
                }
              }
            ]
          }
        ]
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
      throw Exception('Error al obtener descripción: $e');
    }
  }
}
