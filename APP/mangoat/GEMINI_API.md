# 🤖 Integración con Gemini API - MANGOAT

## 📖 Acerca de Gemini

Gemini es el modelo de IA multimodal más avanzado de Google, capaz de comprender y procesar:
- 📝 Texto
- 🖼️ Imágenes
- 🎵 Audio
- 🎥 Video

MANGOAT utiliza **Gemini 2.0 Flash Experimental** para análisis de imágenes de ropa.

## 🔑 API Key

### Obtener tu API Key

1. Visita: [https://aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)
2. Inicia sesión con tu cuenta de Google
3. Haz clic en "Create API Key"
4. Copia tu clave

### Configurar la API Key en MANGOAT

Edita el archivo `lib/config/app_config.dart`:

```dart
static const String geminiApiKey = 'TU_API_KEY_AQUI';
```

## 🚀 Modelo Utilizado

**Gemini 2.0 Flash Experimental**

Características:
- ⚡ Respuestas ultra rápidas (1-3 segundos)
- 🎯 Alta precisión en análisis de imágenes
- 💰 Gratuito hasta 1,500 solicitudes por día
- 📊 Límite de 15 solicitudes por minuto

## 📊 Límites de Uso

### Tier Gratuito
- **Solicitudes por día**: 1,500
- **Solicitudes por minuto**: 15
- **Tokens por minuto**: 1,000,000
- **Tokens por día**: 1,500,000

### Tier de Pago
- Hasta 360 solicitudes por minuto
- Sin límite diario
- Mayor prioridad en la cola

## 🔧 Configuración Actual

```dart
// Modelo
model: "gemini-2.0-flash-exp"

// Parámetros de generación
temperature: 0.4        // Creatividad moderada-baja
topK: 32                // Considera top 32 tokens
topP: 1.0               // Núcleo de probabilidad
maxOutputTokens: 2048   // Máximo de tokens en respuesta
```

### ¿Qué significa cada parámetro?

**Temperature (0.0 - 2.0)**
- 0.0 = Respuestas más determinísticas y consistentes
- 0.4 = Balance entre creatividad y precisión (USADO)
- 2.0 = Respuestas muy creativas y variadas

**topK**
- Limita las opciones de tokens consideradas
- 32 = Buena diversidad sin perder coherencia

**topP**
- Probabilidad acumulativa de tokens
- 1.0 = Considera todos los tokens posibles

**maxOutputTokens**
- Límite de longitud de respuesta
- 2048 = Suficiente para descripciones detalladas

## 📝 Prompts Utilizados

### Análisis de Tags

```
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
```

### Descripción Detallada

```
Describe esta prenda de ropa de manera detallada y atractiva, como si fueras un experto en moda.
Incluye información sobre el estilo, cómo combinarla y para qué ocasiones es ideal.
```

## 🔐 Seguridad

### Buenas Prácticas

✅ **HACER**:
- Mantener la API key privada
- No compartir en repositorios públicos
- Usar variables de entorno en producción
- Monitorear el uso de la API

❌ **NO HACER**:
- Subir la API key a Git
- Compartir la key públicamente
- Dejar la key en el código en producción
- Ignorar límites de uso

### Para Producción

Usa variables de entorno:

```dart
// En lugar de hardcodear
static const String geminiApiKey = String.fromEnvironment('GEMINI_API_KEY');
```

Y ejecuta:
```bash
flutter run --dart-define=GEMINI_API_KEY=tu_key_aqui
```

## 📈 Monitoreo de Uso

Puedes monitorear tu uso en:
[https://aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)

Verás:
- Solicitudes realizadas hoy
- Solicitudes restantes
- Errores y límites alcanzados

## 🐛 Manejo de Errores

### Errores Comunes

**401 Unauthorized**
```
Causa: API key inválida o expirada
Solución: Verifica tu API key en app_config.dart
```

**429 Too Many Requests**
```
Causa: Has excedido el límite de solicitudes
Solución: Espera unos minutos y vuelve a intentar
```

**400 Bad Request**
```
Causa: Formato de imagen no soportado o muy grande
Solución: Verifica que la imagen sea JPG/PNG y < 20MB
```

**500 Internal Server Error**
```
Causa: Error en el servidor de Google
Solución: Reintenta más tarde
```

## 🎨 Formatos de Imagen Soportados

- ✅ JPEG (.jpg, .jpeg)
- ✅ PNG (.png)
- ✅ WebP (.webp)
- ✅ HEIC (.heic)
- ✅ HEIF (.heif)

### Límites
- **Tamaño máximo**: 20 MB por imagen
- **Dimensiones**: Sin límite específico (recomendado < 4K)
- **Cantidad**: Hasta 3,600 imágenes por solicitud

## 🌐 Enlaces Útiles

- [Documentación Oficial](https://ai.google.dev/docs)
- [Gemini API Quickstart](https://ai.google.dev/tutorials/quickstart)
- [Pricing](https://ai.google.dev/pricing)
- [Community](https://discuss.ai.google.dev/)
- [GitHub Examples](https://github.com/google/generative-ai-docs)

## 💡 Optimizaciones Futuras

### Posibles Mejoras

1. **Caché de Resultados**
   - Guardar análisis previos
   - Reducir llamadas a la API

2. **Procesamiento por Lotes**
   - Analizar múltiples prendas a la vez
   - Optimizar costos

3. **Modo Offline**
   - Guardar resultados localmente
   - Sincronizar después

4. **Análisis Avanzado**
   - Detección de marca
   - Estimación de precio
   - Sugerencias de outfits completos

## 📞 Soporte

Si tienes problemas con la API:
1. Revisa la [documentación oficial](https://ai.google.dev/docs)
2. Consulta el [foro de la comunidad](https://discuss.ai.google.dev/)
3. Verifica el [estado del servicio](https://status.cloud.google.com/)

---

**MANGOAT + Gemini = Análisis de moda con IA de última generación 🥭🤖**
