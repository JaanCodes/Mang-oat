/// Modelo de datos para un producto de ropa
class ProductData {
  // Features numéricas básicas
  final double lifeCycleLength;
  final double numStores;
  final double numSizes;
  final double hasPlusSizes;
  final double price;

  // Features categóricas principales
  final int idSeason;
  final String aggregatedFamily;
  final String family;
  final String category;
  final String fabric;
  final String colorName;
  final String lengthType;
  final String silhouetteType;
  final String waistType;
  final String neckLapelType;
  final String sleeveLengthType;
  final String heelShapeType;
  final String toecapType;
  final String wovenStructure;
  final String knitStructure;
  final String printType;
  final String archetype;
  final String moment;

  // Fecha de lanzamiento (phase_in)
  final DateTime? phaseIn;

  // Image embedding (vector de 256 dimensiones)
  final List<double> imageEmbedding;

  ProductData({
    required this.lifeCycleLength,
    required this.numStores,
    required this.numSizes,
    required this.hasPlusSizes,
    required this.price,
    required this.idSeason,
    required this.aggregatedFamily,
    required this.family,
    required this.category,
    required this.fabric,
    required this.colorName,
    required this.lengthType,
    required this.silhouetteType,
    required this.waistType,
    required this.neckLapelType,
    required this.sleeveLengthType,
    required this.heelShapeType,
    required this.toecapType,
    required this.wovenStructure,
    required this.knitStructure,
    required this.printType,
    required this.archetype,
    required this.moment,
    this.phaseIn,
    required this.imageEmbedding,
  });

  /// Crea un ProductData de ejemplo para testing
  factory ProductData.example() {
    return ProductData(
      lifeCycleLength: 12.0,
      numStores: 150.0,
      numSizes: 5.0,
      hasPlusSizes: 1.0,
      price: 29.99,
      idSeason: 202301,
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
      phaseIn: DateTime.now(),
      imageEmbedding: List.filled(256, 0.0),
    );
  }

  /// Crea un ProductData desde JSON (para la API de Gemini)
  factory ProductData.fromJson(Map<String, dynamic> json) {
    // Parsear el embedding si viene como string
    List<double> embedding = [];
    if (json['image_embedding'] != null) {
      if (json['image_embedding'] is String) {
        embedding = (json['image_embedding'] as String)
            .split(',')
            .map((e) => double.tryParse(e.trim()) ?? 0.0)
            .toList();
      } else if (json['image_embedding'] is List) {
        embedding = (json['image_embedding'] as List)
            .map((e) => (e as num).toDouble())
            .toList();
      }
    }

    // Asegurar que el embedding tenga 256 dimensiones
    if (embedding.length < 256) {
      embedding.addAll(List.filled(256 - embedding.length, 0.0));
    } else if (embedding.length > 256) {
      embedding = embedding.sublist(0, 256);
    }

    return ProductData(
      lifeCycleLength: (json['life_cycle_length'] as num?)?.toDouble() ?? 12.0,
      numStores: (json['num_stores'] as num?)?.toDouble() ?? 100.0,
      numSizes: (json['num_sizes'] as num?)?.toDouble() ?? 5.0,
      hasPlusSizes: (json['has_plus_sizes'] as num?)?.toDouble() ?? 0.0,
      price: (json['price'] as num?)?.toDouble() ?? 25.0,
      idSeason: json['id_season'] as int? ?? 202301,
      aggregatedFamily: json['aggregated_family'] as String? ?? 'Woman',
      family: json['family'] as String? ?? 'Unknown',
      category: json['category'] as String? ?? 'Unknown',
      fabric: json['fabric'] as String? ?? 'Unknown',
      colorName: json['color_name'] as String? ?? 'Unknown',
      lengthType: json['length_type'] as String? ?? 'Unknown',
      silhouetteType: json['silhouette_type'] as String? ?? 'Unknown',
      waistType: json['waist_type'] as String? ?? 'Unknown',
      neckLapelType: json['neck_lapel_type'] as String? ?? 'Unknown',
      sleeveLengthType: json['sleeve_length_type'] as String? ?? 'Unknown',
      heelShapeType: json['heel_shape_type'] as String? ?? 'Unknown',
      toecapType: json['toecap_type'] as String? ?? 'Unknown',
      wovenStructure: json['woven_structure'] as String? ?? 'Unknown',
      knitStructure: json['knit_structure'] as String? ?? 'Unknown',
      printType: json['print_type'] as String? ?? 'Unknown',
      archetype: json['archetype'] as String? ?? 'Unknown',
      moment: json['moment'] as String? ?? 'Unknown',
      phaseIn: json['phase_in'] != null
          ? DateTime.tryParse(json['phase_in'] as String)
          : DateTime.now(),
      imageEmbedding: embedding,
    );
  }

  Map<String, dynamic> toJson() {
    return {
      'life_cycle_length': lifeCycleLength,
      'num_stores': numStores,
      'num_sizes': numSizes,
      'has_plus_sizes': hasPlusSizes,
      'price': price,
      'id_season': idSeason,
      'aggregated_family': aggregatedFamily,
      'family': family,
      'category': category,
      'fabric': fabric,
      'color_name': colorName,
      'length_type': lengthType,
      'silhouette_type': silhouetteType,
      'waist_type': waistType,
      'neck_lapel_type': neckLapelType,
      'sleeve_length_type': sleeveLengthType,
      'heel_shape_type': heelShapeType,
      'toecap_type': toecapType,
      'woven_structure': wovenStructure,
      'knit_structure': knitStructure,
      'print_type': printType,
      'archetype': archetype,
      'moment': moment,
      'phase_in': phaseIn?.toIso8601String(),
      'image_embedding': imageEmbedding.join(','),
    };
  }
}
