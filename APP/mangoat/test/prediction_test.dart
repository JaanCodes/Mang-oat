import 'package:flutter_test/flutter_test.dart';
import 'package:mangoat/models/product_data.dart';
import 'package:mangoat/services/preprocessing_service.dart';
import 'package:mangoat/services/prediction_service.dart';

void main() {
  group('Modelo de Predicción de Demanda', () {
    test('Preprocesamiento genera features correctamente', () {
      // Crear producto de ejemplo
      final product = ProductData.example();

      // Procesar el producto
      final features = PreprocessingService.processProduct(product);

      // Verificar que se generaron las features esperadas
      expect(features, isNotNull);
      expect(features['life_cycle_length'], equals(12.0));
      expect(features['month_sin'], isNotNull);
      expect(features['month_cos'], isNotNull);
      expect(features['season'], isNotNull);
      expect(features['sleeve_length_type_X_season'], isNotNull);

      print('✓ Features generadas correctamente:');
      print('  - Numéricas: life_cycle_length, price, month_sin, month_cos');
      print('  - Estación: ${features['season']}');
      print('  - Interacción: ${features['sleeve_length_type_X_season']}');
    });

    test('Features de Fourier calculan correctamente', () {
      // Probar diferentes meses
      final testCases = [
        {'month': 1, 'season': 'Winter'},
        {'month': 6, 'season': 'Summer'},
        {'month': 9, 'season': 'Autumn'},
        {'month': 3, 'season': 'Spring'},
      ];

      for (final testCase in testCases) {
        final date = DateTime(2024, testCase['month'] as int, 15);
        final fourier = PreprocessingService.calculateFourierFeatures(date);
        final season = PreprocessingService.getSeason(date);

        expect(fourier['month_sin'], isNotNull);
        expect(fourier['month_cos'], isNotNull);
        expect(season, equals(testCase['season']));

        print(
          '✓ Mes ${testCase['month']}: ${testCase['season']} - sin=${fourier['month_sin']?.toStringAsFixed(2)}, cos=${fourier['month_cos']?.toStringAsFixed(2)}',
        );
      }
    });

    test('Predicción de demanda devuelve valores razonables', () async {
      final service = PredictionService();
      await service.initialize();

      // Crear varios productos de prueba
      final testProducts = [
        ProductData.example(), // Producto estándar
        ProductData(
          lifeCycleLength: 24.0,
          numStores: 300.0,
          numSizes: 8.0,
          hasPlusSizes: 1.0,
          price: 49.99,
          idSeason: 202401,
          aggregatedFamily: 'Woman',
          family: 'Dresses',
          category: 'Formal',
          fabric: 'Silk',
          colorName: 'Black',
          lengthType: 'Long',
          silhouetteType: 'Fitted',
          waistType: 'Natural',
          neckLapelType: 'V-neck',
          sleeveLengthType: 'Long',
          heelShapeType: 'High',
          toecapType: 'Pointed',
          wovenStructure: 'Satin',
          knitStructure: 'None',
          printType: 'Solid',
          archetype: 'Classic',
          moment: 'Night',
          phaseIn: DateTime(2024, 1, 15),
          imageEmbedding: List.filled(256, 0.2),
        ), // Producto premium
      ];

      for (int i = 0; i < testProducts.length; i++) {
        final result = await service.predictWithDetails(testProducts[i]);

        expect(result.predictedDemand, greaterThan(0));
        expect(result.predictedDemand, lessThan(200000));
        expect(result.confidence, greaterThanOrEqualTo(0));
        expect(result.confidence, lessThanOrEqualTo(100));
        expect(result.demandLevel, isNotNull);

        print('\n✓ Producto ${i + 1}:');
        print(
          '  - Demanda predicha: ${result.predictedDemand.toInt()} unidades/semana',
        );
        print('  - Nivel: ${result.demandLevel}');
        print('  - Confianza: ${result.confidence.toInt()}%');
        print('  - Estación: ${result.season}');
      }
    });

    test('Features de tendencia combinan correctamente', () {
      final product = ProductData(
        lifeCycleLength: 12.0,
        numStores: 150.0,
        numSizes: 5.0,
        hasPlusSizes: 1.0,
        price: 29.99,
        idSeason: 202401,
        aggregatedFamily: 'Woman',
        family: 'Dresses',
        category: 'Casual',
        fabric: 'Cotton',
        colorName: 'Blue',
        lengthType: 'Mid',
        silhouetteType: 'Straight',
        waistType: 'Natural',
        neckLapelType: 'Round',
        sleeveLengthType: 'Long',
        heelShapeType: 'Flat',
        toecapType: 'Round',
        wovenStructure: 'Plain',
        knitStructure: 'Jersey',
        printType: 'Solid',
        archetype: 'Classic',
        moment: 'Day',
        phaseIn: DateTime(2024, 1, 15), // Enero = Winter
        imageEmbedding: List.filled(256, 0.1),
      );

      final features = PreprocessingService.processProduct(product);

      // Verificar que manga larga + invierno se combina correctamente
      expect(features['sleeve_length_type_X_season'], equals('Long_S_Winter'));
      expect(features['family_X_season'], equals('Dresses_S_Winter'));

      print('✓ Features de tendencia generadas:');
      print('  - ${features['sleeve_length_type_X_season']}');
      print('  - ${features['family_X_season']}');
      print('  - ${features['fabric_X_season']}');
      print('  - ${features['length_type_X_season']}');
    });

    test('Similitud coseno funciona correctamente', () {
      final vectorA = [1.0, 0.0, 0.0];
      final vectorB = [1.0, 0.0, 0.0];
      final vectorC = [0.0, 1.0, 0.0];

      final simAB = PreprocessingService.cosineSimilarity(vectorA, vectorB);
      final simAC = PreprocessingService.cosineSimilarity(vectorA, vectorC);

      expect(simAB, closeTo(1.0, 0.0001)); // Vectores idénticos
      expect(simAC, closeTo(0.0, 0.0001)); // Vectores perpendiculares

      print('✓ Similitud coseno:');
      print('  - Vectores idénticos: ${simAB.toStringAsFixed(4)}');
      print('  - Vectores perpendiculares: ${simAC.toStringAsFixed(4)}');
    });

    test('Normalización de features funciona', () {
      final features = {'price': 30.0, 'num_stores': 150.0};

      final means = {'price': 25.0, 'num_stores': 100.0};

      final scales = {'price': 10.0, 'num_stores': 50.0};

      final normalized = PreprocessingService.normalizeNumericFeatures(
        features,
        means,
        scales,
      );

      expect(normalized['price'], closeTo(0.5, 0.0001)); // (30-25)/10
      expect(normalized['num_stores'], closeTo(1.0, 0.0001)); // (150-100)/50

      print('✓ Normalización:');
      print(
        '  - price: ${features['price']} → ${normalized['price']?.toStringAsFixed(2)}',
      );
      print(
        '  - num_stores: ${features['num_stores']} → ${normalized['num_stores']?.toStringAsFixed(2)}',
      );
    });
  });
}
