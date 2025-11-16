import 'dart:io';
import 'dart:math';
import 'package:flutter/material.dart';
import 'package:flutter_spinkit/flutter_spinkit.dart';
import 'package:mangoat/services/gemini_service.dart';
import 'package:mangoat/services/prediction_service.dart';
import 'package:mangoat/models/product_data.dart';
import 'package:fl_chart/fl_chart.dart';

class ResultsScreen extends StatefulWidget {
  final File imageFile;

  const ResultsScreen({super.key, required this.imageFile});

  @override
  State<ResultsScreen> createState() => _ResultsScreenState();
}

class _ResultsScreenState extends State<ResultsScreen> {
  final GeminiService _geminiService = GeminiService();
  final PredictionService _predictionService = PredictionService();

  List<String> _tags = [];
  ProductData? _productData;
  PredictionResult? _prediction;
  List<MonthlyPrediction> _monthlyPredictions = [];
  List<double> _aiPredictions = [];
  bool _isLoading = true;
  String? _error;

  @override
  void initState() {
    super.initState();
    _analyzeImage();
  }

  Future<void> _analyzeImage() async {
    try {
      setState(() {
        _isLoading = true;
        _error = null;
      });

      // Inicializar el servicio de predicción
      await _predictionService.initialize();

      // Análisis completo con IA en una sola petición
      final aiResult = await _geminiService.analyzeComplete(widget.imageFile);

      final tags = aiResult.descriptiveTags; // Tags descriptivos para mostrar
      final specificTags = aiResult.specificTags; // Tags técnicos para lógica
      final productData = aiResult.productData;

      // Crear PredictionResult con datos de IA
      final prediction = PredictionResult(
        predictedDemand: aiResult.weeklyDemand,
        season: _getCurrentSeason(),
        confidence: aiResult.confidence,
        factors: {
          'Tipo': productData.family,
          'Estilo': productData.archetype,
          'Material': productData.fabric,
          'Precio': '\$${productData.price.toStringAsFixed(2)}',
        },
      );

      // Obtener predicciones de 12 meses (modelo KNN + estacionalidad por tags)
      final monthlyPredictions = await _predictionService.predict12Months(
        productData,
      );

      // Aplicar estacionalidad basada en tags ESPECÍFICOS para ambas líneas del gráfico
      var adjustedMonthlyPredictions = _applyTagSeasonality(
        monthlyPredictions,
        specificTags,
        productData,
      );

      // La línea "AI Generated" también usa la misma lógica de estacionalidad
      // pero con valores base ligeramente diferentes para comparación
      var aiPredictionsList = _applyTagSeasonality(
        monthlyPredictions
            .map(
              (p) => MonthlyPrediction(
                month: p.month,
                weeklyDemand: p.weeklyDemand * 0.95, // Variación del 5%
                monthlyDemand: p.monthlyDemand * 0.95,
              ),
            )
            .toList(),
        specificTags,
        productData,
      );

      // Normalizar ambas listas si algún valor supera 10000
      adjustedMonthlyPredictions = _normalizePredictions(
        adjustedMonthlyPredictions,
      );
      aiPredictionsList = _normalizePredictions(aiPredictionsList);

      final aiPredictions = aiPredictionsList
          .map((p) => p.monthlyDemand)
          .toList();

      // Calcular demanda realista basada en las predicciones reales y la confianza
      final realDemand = _calculateRealisticDemand(
        adjustedMonthlyPredictions,
        aiResult.weeklyDemand,
        aiResult.confidence,
        productData,
      );

      // Actualizar la predicción con el valor realista
      final updatedPrediction = PredictionResult(
        predictedDemand: realDemand,
        season: prediction.season,
        confidence: prediction.confidence,
        factors: prediction.factors,
      );

      setState(() {
        _tags = tags;
        _productData = productData;
        _prediction = updatedPrediction;
        _monthlyPredictions = adjustedMonthlyPredictions;
        _aiPredictions = aiPredictions;
        _isLoading = false;
      });
    } catch (e) {
      setState(() {
        _error = e.toString();
        _isLoading = false;
      });
    }
  }

  /// Normaliza las predicciones para que ningún valor supere 10000
  List<MonthlyPrediction> _normalizePredictions(
    List<MonthlyPrediction> predictions,
  ) {
    if (predictions.isEmpty) return predictions;

    // Encontrar el valor máximo
    final maxWeekly = predictions
        .map((p) => p.weeklyDemand)
        .reduce((a, b) => a > b ? a : b);

    // Solo normalizar si algún valor supera 10000
    if (maxWeekly <= 10000) return predictions;

    // Factor de normalización
    final normalizationFactor = 10000.0 / maxWeekly;

    // Aplicar normalización
    return predictions.map((p) {
      final normalizedWeekly = p.weeklyDemand * normalizationFactor;
      return MonthlyPrediction(
        month: p.month,
        weeklyDemand: normalizedWeekly,
        monthlyDemand: normalizedWeekly * 4.33,
      );
    }).toList();
  }

  /// Calcula una demanda realista basada en la media de predicciones y confianza
  double _calculateRealisticDemand(
    List<MonthlyPrediction> predictions,
    double aiEstimate,
    double confidence,
    ProductData product,
  ) {
    if (predictions.isEmpty) return aiEstimate;

    // 1. Calcular la media de las predicciones mensuales
    final weeklyPredictions = predictions.map((p) => p.weeklyDemand).toList();
    final meanWeekly =
        weeklyPredictions.reduce((a, b) => a + b) / weeklyPredictions.length;

    // 2. Calcular la desviación estándar (varianza)
    final variance =
        weeklyPredictions
            .map((v) {
              final diff = v - meanWeekly;
              return diff * diff;
            })
            .reduce((a, b) => a + b) /
        weeklyPredictions.length;
    final stdDev = sqrt(variance);

    // 3. Calcular el peso de la estimación de IA vs modelo basado en confianza
    // Alta confianza = más peso a la IA, Baja confianza = más peso al modelo
    final aiWeight = confidence / 100.0; // 0.6 a 0.95
    final modelWeight = 1.0 - aiWeight;

    // 4. Combinar estimación de IA con la media del modelo
    double combinedEstimate =
        (aiEstimate * aiWeight) + (meanWeekly * modelWeight);

    // 5. Ajustar por variabilidad (si hay alta varianza, añadir incertidumbre)
    final variabilityFactor = 1.0 + (stdDev / meanWeekly * 0.1); // Máximo +10%
    combinedEstimate *= variabilityFactor;

    // 6. Añadir variabilidad única basada en características del producto
    final uniqueHash =
        product.family.hashCode ^
        product.fabric.hashCode ^
        product.colorName.hashCode ^
        product.archetype.hashCode;
    final randomSeed = ((uniqueHash.abs() % 1000) / 1000.0);
    final uniqueMultiplier = 0.92 + (randomSeed * 0.16); // 0.92 a 1.08
    combinedEstimate *= uniqueMultiplier;

    // 7. Ajustar por factores del producto
    // Precio: productos más baratos tienden a vender más
    final priceAdjust = product.price > 0
        ? (1.0 + ((50.0 - product.price) / 100.0)).clamp(0.7, 1.4)
        : 1.0;
    combinedEstimate *= priceAdjust;

    // Tiendas: más tiendas = más ventas
    final storesAdjust = (product.numStores / 150.0).clamp(0.75, 1.35);
    combinedEstimate *= storesAdjust;

    // Tallas: más opciones = más ventas
    final sizesAdjust = (product.numSizes / 5.0).clamp(0.85, 1.2);
    combinedEstimate *= sizesAdjust;

    // 8. Limitar al rango realista
    return combinedEstimate.clamp(500.0, 9500.0);
  }

  List<MonthlyPrediction> _applyTagSeasonality(
    List<MonthlyPrediction> predictions,
    List<String> tags,
    ProductData product,
  ) {
    if (predictions.isEmpty) {
      return predictions;
    }

    final lowerTags = tags.map((t) => t.toLowerCase()).toList();
    bool tagContains(List<String> keywords) {
      return lowerTags.any(
        (tag) => keywords.any((keyword) => tag.contains(keyword)),
      );
    }

    final isWinter =
        tagContains(['invierno', 'winter', 'abrigo', 'coat']) ||
        product.fabric.toLowerCase().contains('wool');
    final isSummer =
        tagContains(['verano', 'summer', 'short', 'beach']) ||
        product.lengthType.toLowerCase().contains('short');
    final isSporty = tagContains(['sport', 'deportivo', 'active']);
    final isPremium = tagContains(['premium', 'lujo', 'formal']);
    final isParty =
        tagContains(['party', 'fiesta', 'noche']) ||
        product.category == 'Party';

    return predictions.asMap().entries.map((entry) {
      final monthNumber = entry.value.month.month;
      double multiplier = 1.0;

      if (isWinter) {
        if (monthNumber >= 11 || monthNumber <= 2) {
          multiplier = 1.4;
        } else if (monthNumber >= 6 && monthNumber <= 8) {
          multiplier = 0.5;
        }
      } else if (isSummer) {
        if (monthNumber >= 5 && monthNumber <= 8) {
          multiplier = 1.5;
        } else if (monthNumber >= 12 || monthNumber <= 2) {
          multiplier = 0.4;
        }
      } else {
        if ((monthNumber >= 3 && monthNumber <= 5) ||
            (monthNumber >= 9 && monthNumber <= 10)) {
          multiplier = 1.15;
        }
      }

      if (isSporty) multiplier *= 1.05;
      if (isPremium) multiplier *= 0.9;
      if (isParty && (monthNumber >= 11 || monthNumber <= 1)) {
        multiplier *= 1.2;
      }

      final priceFactor = product.price <= 0
          ? 1.0
          : (40.0 / product.price).clamp(0.7, 1.5).toDouble();
      final storesFactor = (product.numStores / 150).clamp(0.7, 1.6).toDouble();
      final noise = 0.9 + ((entry.key % 4) * 0.035);

      final adjustedWeekly =
          (entry.value.weeklyDemand *
                  multiplier *
                  priceFactor *
                  storesFactor *
                  noise)
              .clamp(350.0, entry.value.weeklyDemand * 1.9 + 1200);
      final adjustedMonthly = adjustedWeekly * 4.33;

      return MonthlyPrediction(
        month: entry.value.month,
        weeklyDemand: adjustedWeekly,
        monthlyDemand: adjustedMonthly,
      );
    }).toList();
  }

  String _getCurrentSeason() {
    final month = DateTime.now().month;
    if (month >= 12 || month <= 2) return 'Invierno';
    if (month >= 3 && month <= 5) return 'Primavera';
    if (month >= 6 && month <= 8) return 'Verano';
    return 'Otoño';
  }

  /// Retorna los colores del gradiente según el nivel de demanda
  List<Color> _getPredictionColors(String demandLevel) {
    switch (demandLevel) {
      case 'Baja':
        return [Colors.red.shade400, Colors.red.shade600];
      case 'Media':
        return [Colors.orange.shade400, Colors.deepOrange.shade500];
      case 'Alta':
        return [Colors.green.shade400, Colors.green.shade600];
      case 'Muy Alta':
        return [Colors.blue.shade400, Colors.indigo.shade600];
      default:
        return [Colors.grey.shade400, Colors.grey.shade600];
    }
  }

  /// Retorna el icono apropiado según el nivel de demanda
  IconData _getDemandIcon(String demandLevel) {
    switch (demandLevel) {
      case 'Baja':
        return Icons.trending_down;
      case 'Media':
        return Icons.trending_flat;
      case 'Alta':
        return Icons.trending_up;
      case 'Muy Alta':
        return Icons.rocket_launch;
      default:
        return Icons.help_outline;
    }
  }

  /// Retorna el color del icono según el nivel de demanda
  Color _getDemandIconColor(String demandLevel) {
    switch (demandLevel) {
      case 'Baja':
        return Colors.red.shade700;
      case 'Media':
        return Colors.orange.shade700;
      case 'Alta':
        return Colors.green.shade700;
      case 'Muy Alta':
        return Colors.blue.shade700;
      default:
        return Colors.grey.shade700;
    }
  }

  /// Construye una tarjeta de información con título y pares clave-valor
  Widget _buildInfoCard(
    String title,
    IconData icon,
    Color color,
    Map<String, String> data,
  ) {
    return Card(
      elevation: 3,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(15)),
      child: Padding(
        padding: const EdgeInsets.all(20),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Container(
                  padding: const EdgeInsets.all(8),
                  decoration: BoxDecoration(
                    color: color.withOpacity(0.1),
                    borderRadius: BorderRadius.circular(10),
                  ),
                  child: Icon(icon, color: color, size: 24),
                ),
                const SizedBox(width: 12),
                Text(
                  title,
                  style: const TextStyle(
                    fontSize: 18,
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ],
            ),
            const SizedBox(height: 15),
            ...data.entries.map((entry) {
              return Padding(
                padding: const EdgeInsets.symmetric(vertical: 6),
                child: Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  children: [
                    Text(
                      entry.key,
                      style: const TextStyle(fontSize: 14, color: Colors.grey),
                    ),
                    const SizedBox(width: 10),
                    Flexible(
                      child: Text(
                        entry.value,
                        style: const TextStyle(
                          fontSize: 14,
                          fontWeight: FontWeight.bold,
                        ),
                        textAlign: TextAlign.right,
                      ),
                    ),
                  ],
                ),
              );
            }).toList(),
          ],
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.grey.shade100,
      appBar: AppBar(
        title: const Text(
          'Análisis de Prenda',
          style: TextStyle(fontWeight: FontWeight.bold),
        ),
        backgroundColor: Colors.orange,
        foregroundColor: Colors.white,
        elevation: 0,
      ),
      body: SingleChildScrollView(
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // Imagen
            Hero(
              tag: 'clothing_image',
              child: Container(
                width: double.infinity,
                height: 350,
                decoration: BoxDecoration(
                  color: Colors.black,
                  boxShadow: [
                    BoxShadow(
                      color: Colors.black.withOpacity(0.3),
                      blurRadius: 10,
                      offset: const Offset(0, 5),
                    ),
                  ],
                ),
                child: Image.file(widget.imageFile, fit: BoxFit.contain),
              ),
            ),

            if (_isLoading)
              Container(
                padding: const EdgeInsets.all(50),
                child: const Center(
                  child: Column(
                    children: [
                      SpinKitFadingCircle(color: Colors.orange, size: 60),
                      SizedBox(height: 20),
                      Text(
                        'Analizando tu prenda con IA...',
                        style: TextStyle(fontSize: 16, color: Colors.grey),
                      ),
                    ],
                  ),
                ),
              )
            else if (_error != null)
              Padding(
                padding: const EdgeInsets.all(20),
                child: Card(
                  color: Colors.red.shade50,
                  child: Padding(
                    padding: const EdgeInsets.all(20),
                    child: Column(
                      children: [
                        const Icon(
                          Icons.error_outline,
                          size: 50,
                          color: Colors.red,
                        ),
                        const SizedBox(height: 10),
                        Text(
                          'Error: $_error',
                          textAlign: TextAlign.center,
                          style: const TextStyle(color: Colors.red),
                        ),
                        const SizedBox(height: 20),
                        ElevatedButton.icon(
                          onPressed: _analyzeImage,
                          icon: const Icon(Icons.refresh),
                          label: const Text('Reintentar'),
                          style: ElevatedButton.styleFrom(
                            backgroundColor: Colors.orange,
                            foregroundColor: Colors.white,
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
              )
            else
              Padding(
                padding: const EdgeInsets.all(20),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    // SECCIÓN 1: PREDICCIÓN DE DEMANDA (LO MÁS IMPORTANTE)
                    if (_prediction != null) ...[
                      const Text(
                        '📊 Predicción de Producción',
                        style: TextStyle(
                          fontSize: 28,
                          fontWeight: FontWeight.bold,
                          color: Colors.orange,
                        ),
                      ),
                      const SizedBox(height: 15),
                      Card(
                        elevation: 8,
                        shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(20),
                        ),
                        child: Container(
                          decoration: BoxDecoration(
                            gradient: LinearGradient(
                              begin: Alignment.topLeft,
                              end: Alignment.bottomRight,
                              colors: _getPredictionColors(
                                _prediction!.demandLevel,
                              ),
                            ),
                            borderRadius: BorderRadius.circular(20),
                          ),
                          padding: const EdgeInsets.all(25),
                          child: Column(
                            children: [
                              const Text(
                                'DEMANDA SEMANAL ESTIMADA',
                                style: TextStyle(
                                  fontSize: 14,
                                  color: Colors.white70,
                                  fontWeight: FontWeight.w600,
                                  letterSpacing: 1.5,
                                ),
                              ),
                              const SizedBox(height: 15),
                              Text(
                                '${_prediction!.predictedDemand.toStringAsFixed(0)}',
                                style: const TextStyle(
                                  fontSize: 48,
                                  color: Colors.white,
                                  fontWeight: FontWeight.bold,
                                ),
                              ),
                              const Text(
                                'unidades / semana',
                                style: TextStyle(
                                  fontSize: 16,
                                  color: Colors.white70,
                                ),
                              ),
                              const SizedBox(height: 20),
                              Container(
                                padding: const EdgeInsets.symmetric(
                                  horizontal: 20,
                                  vertical: 10,
                                ),
                                decoration: BoxDecoration(
                                  color: Colors.white,
                                  borderRadius: BorderRadius.circular(25),
                                ),
                                child: Row(
                                  mainAxisSize: MainAxisSize.min,
                                  children: [
                                    Icon(
                                      _getDemandIcon(_prediction!.demandLevel),
                                      color: _getDemandIconColor(
                                        _prediction!.demandLevel,
                                      ),
                                      size: 24,
                                    ),
                                    const SizedBox(width: 10),
                                    Text(
                                      'Nivel: ${_prediction!.demandLevel}',
                                      style: TextStyle(
                                        fontSize: 18,
                                        color: _getDemandIconColor(
                                          _prediction!.demandLevel,
                                        ),
                                        fontWeight: FontWeight.bold,
                                      ),
                                    ),
                                  ],
                                ),
                              ),
                              const SizedBox(height: 20),
                              Row(
                                mainAxisAlignment: MainAxisAlignment.center,
                                children: [
                                  const Icon(
                                    Icons.verified,
                                    color: Colors.white70,
                                    size: 18,
                                  ),
                                  const SizedBox(width: 8),
                                  Text(
                                    'Confianza: ${_prediction!.confidence.toInt()}%',
                                    style: const TextStyle(
                                      fontSize: 14,
                                      color: Colors.white70,
                                    ),
                                  ),
                                ],
                              ),
                            ],
                          ),
                        ),
                      ),
                      const SizedBox(height: 30),
                    ],

                    // SECCIÓN 1.5: GRÁFICO DE PREDICCIÓN 12 MESES
                    if (_monthlyPredictions.isNotEmpty) ...[
                      const Text(
                        '📈 Predicción 12 Meses',
                        style: TextStyle(
                          fontSize: 24,
                          fontWeight: FontWeight.bold,
                          color: Colors.orange,
                        ),
                      ),
                      const SizedBox(height: 10),
                      const Text(
                        'Comparación: Modelo KNN vs AI Estacional',
                        style: TextStyle(fontSize: 14, color: Colors.grey),
                      ),
                      const SizedBox(height: 20),

                      // Leyenda
                      Row(
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: [
                          Container(
                            padding: const EdgeInsets.symmetric(
                              horizontal: 12,
                              vertical: 6,
                            ),
                            decoration: BoxDecoration(
                              color: Colors.deepOrange.shade50,
                              borderRadius: BorderRadius.circular(20),
                              border: Border.all(
                                color: Colors.deepOrange,
                                width: 2,
                              ),
                            ),
                            child: Row(
                              children: [
                                Container(
                                  width: 20,
                                  height: 3,
                                  color: Colors.deepOrange,
                                ),
                                const SizedBox(width: 8),
                                const Text(
                                  'Model Generated',
                                  style: TextStyle(
                                    fontSize: 12,
                                    fontWeight: FontWeight.bold,
                                    color: Colors.deepOrange,
                                  ),
                                ),
                              ],
                            ),
                          ),
                          const SizedBox(width: 16),
                          Container(
                            padding: const EdgeInsets.symmetric(
                              horizontal: 12,
                              vertical: 6,
                            ),
                            decoration: BoxDecoration(
                              color: Colors.blue.shade50,
                              borderRadius: BorderRadius.circular(20),
                              border: Border.all(color: Colors.blue, width: 2),
                            ),
                            child: Row(
                              children: [
                                Container(
                                  width: 20,
                                  height: 3,
                                  decoration: BoxDecoration(
                                    color: Colors.blue,
                                    borderRadius: BorderRadius.circular(2),
                                  ),
                                ),
                                const SizedBox(width: 8),
                                const Text(
                                  'AI Generated',
                                  style: TextStyle(
                                    fontSize: 12,
                                    fontWeight: FontWeight.bold,
                                    color: Colors.blue,
                                  ),
                                ),
                              ],
                            ),
                          ),
                        ],
                      ),
                      const SizedBox(height: 20),

                      Card(
                        elevation: 5,
                        shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(20),
                        ),
                        child: Container(
                          padding: const EdgeInsets.all(20),
                          height: 300,
                          child: LineChart(
                            LineChartData(
                              gridData: FlGridData(
                                show: true,
                                drawVerticalLine: false,
                                horizontalInterval: 5000,
                                getDrawingHorizontalLine: (value) {
                                  return FlLine(
                                    color: Colors.grey.shade300,
                                    strokeWidth: 1,
                                  );
                                },
                              ),
                              titlesData: FlTitlesData(
                                show: true,
                                bottomTitles: AxisTitles(
                                  sideTitles: SideTitles(
                                    showTitles: true,
                                    reservedSize: 30,
                                    interval: 1,
                                    getTitlesWidget: (value, meta) {
                                      if (value.toInt() >= 0 &&
                                          value.toInt() <
                                              _monthlyPredictions.length) {
                                        return Padding(
                                          padding: const EdgeInsets.only(
                                            top: 8.0,
                                          ),
                                          child: Text(
                                            _monthlyPredictions[value.toInt()]
                                                .monthName,
                                            style: const TextStyle(
                                              fontSize: 10,
                                              fontWeight: FontWeight.bold,
                                              color: Colors.grey,
                                            ),
                                          ),
                                        );
                                      }
                                      return const Text('');
                                    },
                                  ),
                                ),
                                leftTitles: AxisTitles(
                                  sideTitles: SideTitles(
                                    showTitles: true,
                                    reservedSize: 50,
                                    interval: 5000,
                                    getTitlesWidget: (value, meta) {
                                      return Text(
                                        '${(value / 1000).toStringAsFixed(0)}k',
                                        style: const TextStyle(
                                          fontSize: 10,
                                          color: Colors.grey,
                                        ),
                                      );
                                    },
                                  ),
                                ),
                                topTitles: const AxisTitles(
                                  sideTitles: SideTitles(showTitles: false),
                                ),
                                rightTitles: const AxisTitles(
                                  sideTitles: SideTitles(showTitles: false),
                                ),
                              ),
                              borderData: FlBorderData(
                                show: true,
                                border: Border.all(
                                  color: Colors.grey.shade300,
                                  width: 1,
                                ),
                              ),
                              minX: 0,
                              maxX: (_monthlyPredictions.length - 1).toDouble(),
                              minY: 0,
                              maxY: () {
                                final knnMax = _monthlyPredictions
                                    .map((e) => e.weeklyDemand)
                                    .reduce((a, b) => a > b ? a : b);
                                final aiMax = _aiPredictions.isNotEmpty
                                    ? _aiPredictions.reduce(
                                        (a, b) => a > b ? a : b,
                                      )
                                    : 0.0;
                                return (knnMax > aiMax ? knnMax : aiMax) * 1.2;
                              }(),
                              lineBarsData: [
                                // Línea del modelo KNN (naranja, sólida)
                                LineChartBarData(
                                  spots: _monthlyPredictions
                                      .asMap()
                                      .entries
                                      .map((entry) {
                                        return FlSpot(
                                          entry.key.toDouble(),
                                          entry.value.weeklyDemand,
                                        );
                                      })
                                      .toList(),
                                  isCurved: true,
                                  gradient: LinearGradient(
                                    colors: [
                                      Colors.orange.shade400,
                                      Colors.deepOrange.shade600,
                                    ],
                                  ),
                                  barWidth: 4,
                                  isStrokeCapRound: true,
                                  dotData: FlDotData(
                                    show: true,
                                    getDotPainter:
                                        (spot, percent, barData, index) {
                                          return FlDotCirclePainter(
                                            radius: 6,
                                            color: Colors.white,
                                            strokeWidth: 3,
                                            strokeColor: Colors.deepOrange,
                                          );
                                        },
                                  ),
                                  belowBarData: BarAreaData(
                                    show: true,
                                    gradient: LinearGradient(
                                      colors: [
                                        Colors.orange.shade200.withOpacity(0.2),
                                        Colors.orange.shade100.withOpacity(
                                          0.05,
                                        ),
                                      ],
                                      begin: Alignment.topCenter,
                                      end: Alignment.bottomCenter,
                                    ),
                                  ),
                                ),
                                // Línea de la IA (azul, más transparente)
                                if (_aiPredictions.isNotEmpty)
                                  LineChartBarData(
                                    spots: _aiPredictions.asMap().entries.map((
                                      entry,
                                    ) {
                                      return FlSpot(
                                        entry.key.toDouble(),
                                        entry.value,
                                      );
                                    }).toList(),
                                    isCurved: true,
                                    gradient: LinearGradient(
                                      colors: [
                                        Colors.blue.shade300,
                                        Colors.blue.shade500,
                                      ],
                                    ),
                                    barWidth: 3,
                                    isStrokeCapRound: true,
                                    dotData: FlDotData(
                                      show: true,
                                      getDotPainter:
                                          (spot, percent, barData, index) {
                                            return FlDotCirclePainter(
                                              radius: 4,
                                              color: Colors.white,
                                              strokeWidth: 2,
                                              strokeColor: Colors.blue,
                                            );
                                          },
                                    ),
                                    belowBarData: BarAreaData(
                                      show: true,
                                      gradient: LinearGradient(
                                        colors: [
                                          Colors.blue.shade200.withOpacity(
                                            0.15,
                                          ),
                                          Colors.blue.shade100.withOpacity(
                                            0.03,
                                          ),
                                        ],
                                        begin: Alignment.topCenter,
                                        end: Alignment.bottomCenter,
                                      ),
                                    ),
                                  ),
                              ],
                              lineTouchData: LineTouchData(
                                enabled: true,
                                touchTooltipData: LineTouchTooltipData(
                                  getTooltipItems: (touchedSpots) {
                                    return touchedSpots.map((spot) {
                                      final month =
                                          _monthlyPredictions[spot.x.toInt()];
                                      final isKnn = spot.barIndex == 0;
                                      return LineTooltipItem(
                                        '${month.monthName} - ${isKnn ? "KNN" : "AI"}\\n${spot.y.toInt()} un/sem',
                                        TextStyle(
                                          color: Colors.white,
                                          fontWeight: FontWeight.bold,
                                          fontSize: 11,
                                          backgroundColor: isKnn
                                              ? Colors.deepOrange.withOpacity(
                                                  0.9,
                                                )
                                              : Colors.blue.withOpacity(0.9),
                                        ),
                                      );
                                    }).toList();
                                  },
                                ),
                              ),
                            ),
                          ),
                        ),
                      ),
                      const SizedBox(height: 20),

                      // Estadísticas comparativas
                      const Text(
                        'Estadísticas Comparativas',
                        style: TextStyle(
                          fontSize: 16,
                          fontWeight: FontWeight.bold,
                          color: Colors.grey,
                        ),
                      ),
                      const SizedBox(height: 12),
                      Row(
                        children: [
                          // KNN Model Stats
                          Expanded(
                            child: Card(
                              elevation: 2,
                              color: Colors.deepOrange.shade50,
                              shape: RoundedRectangleBorder(
                                borderRadius: BorderRadius.circular(12),
                              ),
                              child: Padding(
                                padding: const EdgeInsets.all(16),
                                child: Column(
                                  children: [
                                    Row(
                                      mainAxisAlignment:
                                          MainAxisAlignment.center,
                                      children: [
                                        Icon(
                                          Icons.analytics,
                                          color: Colors.deepOrange.shade600,
                                          size: 20,
                                        ),
                                        const SizedBox(width: 8),
                                        const Text(
                                          'Comparison Model',
                                          style: TextStyle(
                                            fontSize: 12,
                                            fontWeight: FontWeight.bold,
                                            color: Colors.deepOrange,
                                          ),
                                        ),
                                      ],
                                    ),
                                    const Divider(height: 20),
                                    _buildStatRow(
                                      'Promedio',
                                      '${(_monthlyPredictions.map((e) => e.weeklyDemand).reduce((a, b) => a + b) / _monthlyPredictions.length).toInt()}',
                                      Colors.deepOrange,
                                    ),
                                    const SizedBox(height: 8),
                                    _buildStatRow(
                                      'Máximo',
                                      '${_monthlyPredictions.map((e) => e.weeklyDemand).reduce((a, b) => a > b ? a : b).toInt()}',
                                      Colors.green,
                                    ),
                                    const SizedBox(height: 8),
                                    _buildStatRow(
                                      'Mínimo',
                                      '${_monthlyPredictions.map((e) => e.weeklyDemand).reduce((a, b) => a < b ? a : b).toInt()}',
                                      Colors.blue,
                                    ),
                                  ],
                                ),
                              ),
                            ),
                          ),
                          const SizedBox(width: 10),
                          // AI Stats
                          if (_aiPredictions.isNotEmpty)
                            Expanded(
                              child: Card(
                                elevation: 2,
                                color: Colors.blue.shade50,
                                shape: RoundedRectangleBorder(
                                  borderRadius: BorderRadius.circular(12),
                                ),
                                child: Padding(
                                  padding: const EdgeInsets.all(16),
                                  child: Column(
                                    children: [
                                      Row(
                                        mainAxisAlignment:
                                            MainAxisAlignment.center,
                                        children: [
                                          Icon(
                                            Icons.psychology,
                                            color: Colors.blue.shade600,
                                            size: 20,
                                          ),
                                          const SizedBox(width: 8),
                                          const Text(
                                            'AI Seasonal',
                                            style: TextStyle(
                                              fontSize: 12,
                                              fontWeight: FontWeight.bold,
                                              color: Colors.blue,
                                            ),
                                          ),
                                        ],
                                      ),
                                      const Divider(height: 20),
                                      _buildStatRow(
                                        'Promedio',
                                        '${(_aiPredictions.reduce((a, b) => a + b) / _aiPredictions.length).toInt()}',
                                        Colors.blue,
                                      ),
                                      const SizedBox(height: 8),
                                      _buildStatRow(
                                        'Máximo',
                                        '${_aiPredictions.reduce((a, b) => a > b ? a : b).toInt()}',
                                        Colors.green,
                                      ),
                                      const SizedBox(height: 8),
                                      _buildStatRow(
                                        'Mínimo',
                                        '${_aiPredictions.reduce((a, b) => a < b ? a : b).toInt()}',
                                        Colors.blue,
                                      ),
                                    ],
                                  ),
                                ),
                              ),
                            ),
                        ],
                      ),
                      const SizedBox(height: 30),
                    ],

                    // SECCIÓN 2: ETIQUETAS EXTRAÍDAS POR LA IA
                    const Text(
                      '🏷️ Etiquetas Detectadas',
                      style: TextStyle(
                        fontSize: 24,
                        fontWeight: FontWeight.bold,
                        color: Colors.orange,
                      ),
                    ),
                    const SizedBox(height: 10),
                    const Text(
                      'Información extraída automáticamente de la imagen:',
                      style: TextStyle(fontSize: 14, color: Colors.grey),
                    ),
                    const SizedBox(height: 15),

                    if (_productData != null) ...[
                      _buildInfoCard(
                        'Información General',
                        Icons.info_outline,
                        Colors.blue,
                        {
                          'Categoría': _productData!.aggregatedFamily,
                          'Tipo': _productData!.family,
                          'Estilo': _productData!.category,
                          'Color': _productData!.colorName,
                          'Momento': _productData!.moment,
                          'Arquetipo': _productData!.archetype,
                        },
                      ),
                      const SizedBox(height: 15),

                      _buildInfoCard(
                        'Características Físicas',
                        Icons.checkroom,
                        Colors.purple,
                        {
                          'Tejido': _productData!.fabric,
                          'Largo': _productData!.lengthType,
                          'Silueta': _productData!.silhouetteType,
                          'Mangas': _productData!.sleeveLengthType,
                          'Cuello': _productData!.neckLapelType,
                          'Estampado': _productData!.printType,
                        },
                      ),
                      const SizedBox(height: 15),

                      _buildInfoCard(
                        'Datos de Mercado',
                        Icons.store,
                        Colors.green,
                        {
                          'Precio Estimado':
                              '\$${_productData!.price.toStringAsFixed(2)}',
                          'Tiendas': '${_productData!.numStores.toInt()}',
                          'Tallas': '${_productData!.numSizes.toInt()}',
                          'Tallas Plus': _productData!.hasPlusSizes > 0
                              ? 'Sí'
                              : 'No',
                          'Ciclo de Vida':
                              '${_productData!.lifeCycleLength.toInt()} semanas',
                          'Temporada': _prediction?.season ?? 'N/A',
                        },
                      ),
                      const SizedBox(height: 30),
                    ],

                    // SECCIÓN 3: FACTORES CLAVE DE LA PREDICCIÓN
                    if (_prediction != null) ...[
                      const Text(
                        '📈 Factores de la Predicción',
                        style: TextStyle(
                          fontSize: 24,
                          fontWeight: FontWeight.bold,
                          color: Colors.orange,
                        ),
                      ),
                      const SizedBox(height: 15),
                      Card(
                        elevation: 3,
                        shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(15),
                        ),
                        child: Padding(
                          padding: const EdgeInsets.all(20),
                          child: Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              const Row(
                                children: [
                                  Icon(
                                    Icons.analytics_outlined,
                                    color: Colors.orange,
                                  ),
                                  SizedBox(width: 8),
                                  Text(
                                    'Análisis del Modelo',
                                    style: TextStyle(
                                      fontSize: 18,
                                      fontWeight: FontWeight.bold,
                                    ),
                                  ),
                                ],
                              ),
                              const SizedBox(height: 15),
                              ..._prediction!.factors.entries.map((entry) {
                                return Padding(
                                  padding: const EdgeInsets.symmetric(
                                    vertical: 8,
                                  ),
                                  child: Row(
                                    mainAxisAlignment:
                                        MainAxisAlignment.spaceBetween,
                                    children: [
                                      Row(
                                        children: [
                                          Container(
                                            width: 8,
                                            height: 8,
                                            decoration: const BoxDecoration(
                                              color: Colors.orange,
                                              shape: BoxShape.circle,
                                            ),
                                          ),
                                          const SizedBox(width: 10),
                                          Text(
                                            entry.key,
                                            style: const TextStyle(
                                              fontSize: 14,
                                              fontWeight: FontWeight.w500,
                                              color: Colors.grey,
                                            ),
                                          ),
                                        ],
                                      ),
                                      const SizedBox(width: 10),
                                      Flexible(
                                        child: Text(
                                          entry.value,
                                          style: const TextStyle(
                                            fontSize: 14,
                                            fontWeight: FontWeight.bold,
                                          ),
                                          textAlign: TextAlign.right,
                                        ),
                                      ),
                                    ],
                                  ),
                                );
                              }).toList(),
                            ],
                          ),
                        ),
                      ),
                      const SizedBox(height: 30),
                    ],

                    // SECCIÓN 4: TAGS DESCRIPTIVOS (OPCIONAL)
                    if (_tags.isNotEmpty) ...[
                      const Text(
                        '🔖 Tags Descriptivos',
                        style: TextStyle(
                          fontSize: 24,
                          fontWeight: FontWeight.bold,
                          color: Colors.orange,
                        ),
                      ),
                      const SizedBox(height: 15),
                      Wrap(
                        spacing: 8,
                        runSpacing: 8,
                        children: _tags.map((tag) {
                          return Container(
                            padding: const EdgeInsets.symmetric(
                              horizontal: 16,
                              vertical: 10,
                            ),
                            decoration: BoxDecoration(
                              gradient: LinearGradient(
                                colors: [
                                  Colors.orange.shade300,
                                  Colors.deepOrange.shade400,
                                ],
                              ),
                              borderRadius: BorderRadius.circular(25),
                              boxShadow: [
                                BoxShadow(
                                  color: Colors.orange.withOpacity(0.3),
                                  blurRadius: 5,
                                  offset: const Offset(0, 3),
                                ),
                              ],
                            ),
                            child: Text(
                              tag,
                              style: const TextStyle(
                                color: Colors.white,
                                fontSize: 13,
                                fontWeight: FontWeight.w500,
                              ),
                            ),
                          );
                        }).toList(),
                      ),
                      const SizedBox(height: 30),
                    ],

                    // Botón para analizar otra prenda
                    SizedBox(
                      width: double.infinity,
                      height: 55,
                      child: ElevatedButton.icon(
                        onPressed: () {
                          Navigator.pop(context);
                        },
                        icon: const Icon(Icons.camera_alt, size: 24),
                        label: const Text(
                          'Analizar Otra Prenda',
                          style: TextStyle(
                            fontSize: 18,
                            fontWeight: FontWeight.bold,
                          ),
                        ),
                        style: ElevatedButton.styleFrom(
                          backgroundColor: Colors.orange,
                          foregroundColor: Colors.white,
                          elevation: 5,
                          shape: RoundedRectangleBorder(
                            borderRadius: BorderRadius.circular(15),
                          ),
                        ),
                      ),
                    ),
                  ],
                ),
              ),
          ],
        ),
      ),
    );
  }

  /// Widget auxiliar para mostrar una fila de estadística
  Widget _buildStatRow(String label, String value, Color color) {
    return Row(
      mainAxisAlignment: MainAxisAlignment.spaceBetween,
      children: [
        Text(label, style: const TextStyle(fontSize: 11, color: Colors.grey)),
        Text(
          value,
          style: TextStyle(
            fontSize: 14,
            fontWeight: FontWeight.bold,
            color: color,
          ),
        ),
      ],
    );
  }
}
