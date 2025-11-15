# 🥭 MANGOAT - Análisis de Ropa con IA

Una aplicación Flutter hermosa y moderna que utiliza la API de SambaNova con el modelo Llama-4-Maverick para analizar prendas de ropa y generar tags descriptivos automáticamente.

## 🌟 Características

- 📸 **Captura de fotos** directamente desde la cámara
- 🖼️ **Selección de imágenes** desde la galería
- 🤖 **Análisis con IA** utilizando Llama-4-Maverick-17B-128E-Instruct
- 🏷️ **Generación automática de tags** descriptivos
- 📝 **Descripción detallada** de la prenda con recomendaciones de estilo
- 🎨 **Interfaz moderna** con gradientes y animaciones

## 📋 Requisitos Previos

- Flutter SDK (3.9.2 o superior)
- Dart SDK
- Android Studio o Xcode (para emuladores)
- Una API Key de SambaNova AI

## 🚀 Instalación

1. **Clonar el repositorio**
   ```bash
   git clone <tu-repositorio>
   cd mangoat
   ```

2. **Instalar dependencias**
   ```bash
   flutter pub get
   ```

3. **Configurar la API Key**
   La API Key de SambaNova ya está configurada en `lib/services/gemini_service.dart`:
   ```dart
   static const String _apiKey = '65cdb83d-db28-42be-8b7c-03d63132c62d';
   static const String _model = 'Llama-4-Maverick-17B-128E-Instruct';
   ```

4. **Ejecutar la aplicación**
   ```bash
   flutter run
   ```

## 📱 Uso

1. Abre la aplicación MANGOAT
2. Elige una opción:
   - **Tomar Foto**: Captura una imagen con la cámara
   - **Elegir de Galería**: Selecciona una imagen existente
3. Espera mientras la IA analiza la prenda
4. Visualiza los tags generados y la descripción detallada
5. ¡Analiza otra prenda cuando quieras!

## 🛠️ Tecnologías Utilizadas

- **Flutter**: Framework de desarrollo multiplataforma
- **SambaNova AI**: API de inteligencia artificial con modelo Llama-4-Maverick
- **Llama-4-Maverick-17B-128E-Instruct**: Modelo de IA multimodal para análisis de imágenes
- **camera**: Plugin para acceso a la cámara del dispositivo
- **image_picker**: Selección de imágenes de la galería
- **http**: Cliente HTTP para comunicación con la API
- **flutter_spinkit**: Indicadores de carga animados
- **permission_handler**: Gestión de permisos de la app

## 📦 Dependencias Principales

```yaml
dependencies:
  flutter:
    sdk: flutter
  camera: ^0.11.0
  permission_handler: ^11.3.1
  http: ^1.2.1
  image: ^4.1.7
  image_picker: ^1.0.7
  flutter_spinkit: ^5.2.0
  path_provider: ^2.1.2
```

## 🔒 Permisos

### Android
- `CAMERA`: Para tomar fotos
- `INTERNET`: Para comunicarse con la API de SambaNova
- `READ_EXTERNAL_STORAGE`: Para acceder a la galería
- `WRITE_EXTERNAL_STORAGE`: Para guardar imágenes temporales

### iOS
- `NSCameraUsageDescription`: Acceso a la cámara
- `NSPhotoLibraryUsageDescription`: Acceso a la galería
- `NSMicrophoneUsageDescription`: Para usar la cámara

## 🎨 Estructura del Proyecto

```
lib/
├── main.dart                    # Punto de entrada de la app
├── screens/
│   ├── home_screen.dart        # Pantalla principal con opciones de captura
│   └── results_screen.dart     # Pantalla de resultados con tags
└── services/
    └── gemini_service.dart     # Servicio para comunicación con Gemini API
```

## 🌈 Características de la Interfaz

- **Gradientes vibrantes**: Naranja, rojo y rosa
- **Animaciones suaves**: Elementos con transiciones animadas
- **Cards con sombras**: Diseño moderno con elevación
- **Tags coloridos**: Visualización atractiva de las etiquetas
- **Indicadores de carga**: Feedback visual durante el procesamiento

## 🔧 Configuración Adicional

### Para Android
Los permisos ya están configurados en `android/app/src/main/AndroidManifest.xml`

### Para iOS
Los permisos ya están configurados en `ios/Runner/Info.plist`

## 📸 Capturas de Pantalla

La aplicación incluye:
- Una pantalla de inicio con gradiente vibrante y logo animado
- Botones grandes y accesibles para tomar foto o elegir de galería
- Pantalla de resultados con la imagen, tags coloridos y descripción detallada

## 🤝 Contribuir

Las contribuciones son bienvenidas. Por favor:
1. Haz fork del proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📄 Licencia

Este proyecto es parte del Datathon 2025.

## 👥 Autores

Desarrollado para el Datathon 2025

## 🙏 Agradecimientos

- SambaNova AI por proporcionar la API con el modelo Llama-4-Maverick
- Flutter team por el excelente framework
- Comunidad de Flutter por los plugins utilizados

---

**¡Disfruta analizando tu ropa con MANGOAT! 🥭👕**

