import 'package:flutter/material.dart';

/// Tema y colores personalizados de la aplicación MANGOAT
class AppTheme {
  // Colores principales
  static const Color primaryOrange = Color(0xFFFF9800);
  static const Color deepOrange = Color(0xFFFF5722);
  static const Color pinkAccent = Color(0xFFEC407A);
  static const Color white = Color(0xFFFFFFFF);
  static const Color black = Color(0xFF000000);
  static const Color greyBackground = Color(0xFFF5F5F5);
  
  // Gradientes
  static const LinearGradient primaryGradient = LinearGradient(
    begin: Alignment.topLeft,
    end: Alignment.bottomRight,
    colors: [
      Color(0xFFFF9800), // Orange
      Color(0xFFFF5722), // Deep Orange
      Color(0xFFEC407A), // Pink
    ],
  );
  
  static const LinearGradient buttonGradient = LinearGradient(
    begin: Alignment.centerLeft,
    end: Alignment.centerRight,
    colors: [
      Color(0xFFFF9800),
      Color(0xFFFF5722),
    ],
  );
  
  // Sombras
  static List<BoxShadow> cardShadow = [
    BoxShadow(
      color: Colors.black.withOpacity(0.1),
      blurRadius: 10,
      offset: const Offset(0, 5),
    ),
  ];
  
  static List<BoxShadow> buttonShadow = [
    BoxShadow(
      color: primaryOrange.withOpacity(0.3),
      blurRadius: 8,
      offset: const Offset(0, 4),
    ),
  ];
  
  // Estilos de texto
  static const TextStyle appBarTitle = TextStyle(
    fontSize: 20,
    fontWeight: FontWeight.bold,
    color: white,
  );
  
  static const TextStyle titleLarge = TextStyle(
    fontSize: 56,
    fontWeight: FontWeight.bold,
    color: white,
    letterSpacing: 4,
  );
  
  static const TextStyle subtitle = TextStyle(
    fontSize: 18,
    color: white,
    fontWeight: FontWeight.w300,
  );
  
  static const TextStyle sectionTitle = TextStyle(
    fontSize: 24,
    fontWeight: FontWeight.bold,
    color: primaryOrange,
  );
  
  static const TextStyle tagText = TextStyle(
    color: white,
    fontSize: 14,
    fontWeight: FontWeight.w500,
  );
  
  static const TextStyle bodyText = TextStyle(
    fontSize: 16,
    height: 1.6,
    color: Color(0xFF212121),
  );
  
  static const TextStyle buttonText = TextStyle(
    fontSize: 20,
    fontWeight: FontWeight.bold,
  );
  
  // Border Radius
  static BorderRadius cardRadius = BorderRadius.circular(15);
  static BorderRadius buttonRadius = BorderRadius.circular(20);
  static BorderRadius tagRadius = BorderRadius.circular(25);
  
  // Tema de la aplicación
  static ThemeData get lightTheme {
    return ThemeData(
      useMaterial3: true,
      colorScheme: ColorScheme.fromSeed(
        seedColor: primaryOrange,
        primary: primaryOrange,
        secondary: deepOrange,
      ),
      fontFamily: 'Roboto',
      scaffoldBackgroundColor: greyBackground,
      appBarTheme: const AppBarTheme(
        backgroundColor: primaryOrange,
        foregroundColor: white,
        elevation: 0,
        centerTitle: true,
        titleTextStyle: appBarTitle,
      ),
      elevatedButtonTheme: ElevatedButtonThemeData(
        style: ElevatedButton.styleFrom(
          backgroundColor: primaryOrange,
          foregroundColor: white,
          elevation: 5,
          padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 16),
          shape: RoundedRectangleBorder(
            borderRadius: buttonRadius,
          ),
          textStyle: buttonText,
        ),
      ),
      cardTheme: CardThemeData(
        elevation: 3,
        shape: RoundedRectangleBorder(
          borderRadius: cardRadius,
        ),
      ),
    );
  }
}
