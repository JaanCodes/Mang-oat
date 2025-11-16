import 'dart:io';
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

      // Obtener tags, descripción y datos del producto en paralelo
      final results = await Future.wait([
        _geminiService.analyzeClothing(widget.imageFile),
        _geminiService.extractProductData(widget.imageFile),
      ]);

      final productData = results[1] as ProductData;

      // Hacer la predicción de demanda
      final prediction = await _predictionService.predictWithDetails(
        productData,
      );

      // Obtener predicciones de 12 meses (modelo KNN)
      final monthlyPredictions = await _predictionService.predict12Months(
        productData,
      );

      // Obtener predicciones de IA (LLM con estacionalidad)
      final aiPredictions = await _geminiService.predictMonthlyDemandWithAI(
        productData,
      );

      setState(() {
        _tags = results[0] as List<String>;
        _productData = productData;
        _prediction = prediction;
        _monthlyPredictions = monthlyPredictions;
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
