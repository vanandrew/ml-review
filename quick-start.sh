#!/bin/bash

# Quick Start Script for Firebase Authentication Testing
# Run this after setting up Firebase config

echo "🚀 ML Review - Firebase Auth Quick Start"
echo "========================================"
echo ""

# Check if .env.local exists
if [ ! -f ".env.local" ]; then
    echo "❌ Error: .env.local not found!"
    echo ""
    echo "Please create .env.local with your Firebase credentials:"
    echo ""
    cat << EOF
VITE_FIREBASE_API_KEY=your_api_key_here
VITE_FIREBASE_AUTH_DOMAIN=your_project.firebaseapp.com
VITE_FIREBASE_PROJECT_ID=your_project_id
VITE_FIREBASE_STORAGE_BUCKET=your_project.appspot.com
VITE_FIREBASE_MESSAGING_SENDER_ID=your_sender_id
VITE_FIREBASE_APP_ID=your_app_id
VITE_FIREBASE_MEASUREMENT_ID=your_measurement_id
EOF
    echo ""
    exit 1
fi

echo "✅ .env.local found"
echo ""

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo "📦 Installing dependencies..."
    npm install
    echo ""
fi

echo "✅ Dependencies installed"
echo ""

# Build the app
echo "🔨 Building app..."
npm run build

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Build successful!"
    echo ""
    echo "🎉 Everything is ready!"
    echo ""
    echo "Next steps:"
    echo "1. Start dev server: npm run dev"
    echo "2. Open http://localhost:5173"
    echo "3. Click 'Sign Up' in sidebar"
    echo "4. Create an account and test!"
    echo ""
    echo "📚 Documentation:"
    echo "  - IMPLEMENTATION_SUMMARY.md - Quick overview"
    echo "  - FIREBASE_AUTH.md - Technical details"
    echo "  - SETUP_CHECKLIST.md - Testing checklist"
    echo ""
    echo "🚀 Ready to start? Run: npm run dev"
    echo ""
else
    echo ""
    echo "❌ Build failed!"
    echo "Check the error messages above."
    echo ""
    exit 1
fi
