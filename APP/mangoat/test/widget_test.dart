// This is a basic Flutter widget test.
//
// To perform an interaction with a widget in your test, use the WidgetTester
// utility in the flutter_test package. For example, you can send tap and scroll
// gestures. You can also use WidgetTester to find child widgets in the widget
// tree, read text, and verify that the values of widget properties are correct.

import 'package:flutter_test/flutter_test.dart';

import 'package:mangoat/main.dart';

void main() {
  testWidgets('MANGOAT app smoke test', (WidgetTester tester) async {
    // Build our app and trigger a frame.
    await tester.pumpWidget(const MangoatApp());

    // Verify that the app title is displayed.
    expect(find.text('MANGOAT'), findsWidgets);
    
    // Verify that the main buttons are present.
    expect(find.text('Tomar Foto'), findsOneWidget);
    expect(find.text('Elegir de Galería'), findsOneWidget);
  });
}
