# 🚀 Guía Rápida de Inicio - MANGOAT

## ⚡ Inicio Rápido

### 1. Instalar las dependencias
```bash
flutter pub get
```

### 2. Ejecutar la aplicación
```bash
flutter run
```

O selecciona el dispositivo específico:
```bash
flutter run -d windows    # Para Windows
flutter run -d chrome     # Para Web
flutter run -d android    # Para Android
flutter run -d ios        # Para iOS
```

## 📱 Dispositivos Soportados

- ✅ Android (API 21+)
- ✅ iOS (12.0+)
- ✅ Web
- ✅ Windows
- ✅ macOS
- ✅ Linux

## 🔑 Configuración de la API Key

La API Key de Gemini ya está configurada en el archivo:
```
lib/services/gemini_service.dart
```

Si necesitas cambiarla:
```dart
static const String _apiKey = 'TU_NUEVA_API_KEY';
```

## 🧪 Ejecutar Tests

```bash
flutter test
```

## 🏗️ Compilar para Producción

### Android (APK)
```bash
flutter build apk --release
```

### Android (App Bundle)
```bash
flutter build appbundle --release
```

### iOS
```bash
flutter build ios --release
```

### Web
```bash
flutter build web --release
```

### Windows
```bash
flutter build windows --release
```

## 🎯 Estructura de Archivos Creados

```
lib/
├── main.dart                          # Punto de entrada
├── config/
│   └── app_config.dart               # Configuración general
├── theme/
│   └── app_theme.dart                # Tema y colores
├── screens/
│   ├── home_screen.dart              # Pantalla principal
│   └── results_screen.dart           # Pantalla de resultados
├── services/
│   └── gemini_service.dart           # Servicio de IA
└── widgets/
    └── tips_card.dart                # Widget de consejos
```

## 📦 Dependencias Instaladas

```yaml
✅ camera: ^0.11.0                     # Captura de fotos
✅ permission_handler: ^11.3.1        # Permisos
✅ http: ^1.2.1                        # Peticiones HTTP
✅ image: ^4.1.7                       # Procesamiento de imágenes
✅ image_picker: ^1.0.7                # Selector de imágenes
✅ flutter_spinkit: ^5.2.0             # Indicadores de carga
✅ path_provider: ^2.1.2               # Rutas del sistema
```

## 🐛 Solución de Problemas Comunes

### Error de permisos en Android
```bash
flutter clean
flutter pub get
```

### Error de cámara en iOS
Verifica que los permisos estén en `ios/Runner/Info.plist`

### Error de compilación
```bash
flutter clean
flutter pub cache repair
flutter pub get
```

## 💡 Consejos de Desarrollo

1. **Hot Reload**: Presiona `r` en la terminal para recargar
2. **Hot Restart**: Presiona `R` para reiniciar
3. **Ver logs**: `flutter logs`
4. **Análisis de código**: `flutter analyze`

## 🎨 Personalización

### Cambiar colores
Edita `lib/theme/app_theme.dart`:
```dart
static const Color primaryOrange = Color(0xFFFF9800);
```

### Cambiar textos
Edita los archivos en `lib/screens/`

### Cambiar comportamiento de la IA
Edita `lib/services/gemini_service.dart` y modifica los prompts

## 📸 Uso de la App

1. **Abrir la app** → Pantalla con gradiente naranja
2. **Tomar Foto** → Captura con la cámara
3. **Elegir de Galería** → Selecciona una imagen existente
4. **Esperar análisis** → La IA procesa la imagen (2-5 segundos)
5. **Ver resultados** → Tags + descripción detallada

## 🔄 Actualizar Dependencias

```bash
flutter pub upgrade
```

## 📊 Verificar Estado del Proyecto

```bash
flutter doctor -v
```

---

**¿Problemas? Revisa la documentación completa en README.md**
