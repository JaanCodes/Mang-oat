/// Constantes de configuración de la aplicación
class AppConfig {
  // SambaNova API Configuration
  static const String apiKey = '65cdb83d-db28-42be-8b7c-03d63132c62d';
  static const String model = 'Llama-4-Maverick-17B-128E-Instruct';
  static const String baseUrl = 'https://api.sambanova.ai/v1/chat/completions';
  
  // App Information
  static const String appName = 'MANGOAT';
  static const String appVersion = '1.0.0';
  
  // Image Configuration
  static const int maxImageWidth = 1920;
  static const int maxImageHeight = 1080;
  static const int imageQuality = 85;
  
  // UI Configuration
  static const Duration animationDuration = Duration(milliseconds: 1500);
  static const Duration loadingMinDuration = Duration(seconds: 2);
}
