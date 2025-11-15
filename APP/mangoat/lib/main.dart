import 'package:flutter/material.dart';
import 'package:mangoat/screens/home_screen.dart';
import 'package:mangoat/theme/app_theme.dart';

void main() {
  runApp(const MangoatApp());
}

class MangoatApp extends StatelessWidget {
  const MangoatApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'MANGOAT',
      debugShowCheckedModeBanner: false,
      theme: AppTheme.lightTheme,
      home: const HomeScreen(),
    );
  }
}
